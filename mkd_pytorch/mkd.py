import os
import numpy as np
import torch
import torch.nn as nn

from .layers import Gradients, EmbedGradients, ExplicitSpacialEncoding, L2Norm, Whitening


class MKD(nn.Module):
    """
    Module, which computes Multiple kernel local descriptors of given patches.
    Args:
        dtype: (str) Parametrization of kernel.
              'concat', 'polar', 'cart'('concat' is default)
        patch_size: (int) Input patch size in pixels (32 is default)
        whitening: (str) Whitening transform to use.
              None, 'lw', 'pca', 'pcaws', 'pcawt'('pcawt' is default)
        training_set: (str) Dataset from PhotoTourism on which whitening is trained.
              'liberty', 'notredame', 'yosemite'()
        reduce_dims: (int) Dimensionality reduction (128 is default).
        do_l2: (bool) l2-normalize individual embeddings (True is default).
        do_final_l2: (bool) l2-normalize final embedding (True is default).
        do_gmask: (bool) Apply gaussian mask to centre (True is default).
        device: (str) Torch device to use ('cpu' is default).
    Returns:
        Tensor: MKD descriptor of the patches.
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, out_dim)
    Examples::
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> mkd = mkd_pytorch.MKD(dtype='concat',
                                  patch_size=32,
                                  whitening='pcawt',
                                  training_set='liberty',
                                  device='cpu')
        >>> descs = mkd(patches) # 23x128
    """

    def __init__(self,
        dtype: str = 'concat',
        patch_size: int = 32,
        whitening: str = 'pcawt',
        training_set: str = 'liberty',
        reduce_dims: int = 128,
        do_l2: bool = True,
        do_final_l2: bool = True,
        do_gmask: bool = True,
        device: str = 'cpu') -> None:
        super().__init__()

        self.patch_size = patch_size
        self.whitening = whitening
        self.reduce_dims = reduce_dims
        self.training_set = training_set
        self.do_l2 = do_l2
        self.do_final_l2 = do_final_l2
        self.do_gmask = do_gmask
        self.device = device
        self.in_shape = [-1, 1, patch_size, patch_size]
        self.dtype = dtype
        self.norm = L2Norm()

        # Use the correct model_file.
        this_dir, _ = os.path.split(__file__)
        self.model_file = os.path.join(this_dir, f'mkd-{dtype}-64.pkl')

        self.grads = Gradients(patch_size=patch_size,
                               do_smoothing=True,
                               sigma=1.4 * (patch_size / 64.0),
                               device=device)

        # Cartesian embedding with absolute gradients.
        if dtype in ['cart', 'concat']:
            ori_abs = EmbedGradients(patch_size=patch_size,
                                     relative=False,
                                     device=device)
            cart_emb = ExplicitSpacialEncoding(dtype='cart',
                                               fmap_size=self.patch_size,
                                               in_dims=7,
                                               do_gmask=self.do_gmask,
                                               do_l2=self.do_l2)
            self.cart_feats = nn.Sequential(ori_abs, cart_emb)

        # Polar embedding with relative gradients.
        if dtype in ['polar', 'concat']:
            ori_rel = EmbedGradients(patch_size=patch_size,
                                     relative=True,
                                     device=device)
            polar_emb = ExplicitSpacialEncoding(dtype='polar',
                                               fmap_size=self.patch_size,
                                               in_dims=7,
                                               do_gmask=self.do_gmask,
                                               do_l2=self.do_l2)
            self.polar_feats = nn.Sequential(ori_rel, polar_emb)

        if dtype == 'concat':
            self.odims = polar_emb.odims + cart_emb.odims
        elif dtype == 'cart':
            self.odims = cart_emb.odims
        elif dtype == 'polar':
            self.odims = polar_emb.odims

        # Redundancy to support old code somewhere.
        self.out_dim = self.odims

        # Load supervised (lw) or unsupervised (pca) model trained on training_set.
        if self.whitening is not None:
            algo = 'lw' if self.whitening == 'lw' else 'pca'
            whitening_models = np.load(self.model_file, allow_pickle=True)
            whitening_model = whitening_models[training_set][algo]
            self.whitening_layer = Whitening(whitening,
                                             whitening_model,
                                             reduce_dims=reduce_dims,
                                             device=device)

            self.out_dim = self.reduce_dims
            self.odims = self.reduce_dims

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        g = self.grads(patches)

        if self.dtype in ['polar', 'concat']:
            pe = self.polar_feats(g)
        if self.dtype in ['cart', 'concat']:
            ce = self.cart_feats(g)

        if self.dtype == 'concat':
            y = torch.cat([pe, ce], dim=1)
        elif self.dtype == 'cart':
            y = ce
        elif self.dtype == 'polar':
            y = pe
        if self.do_final_l2:
            y = self.norm(y)

        if self.whitening is not None:
            y = self.whitening_layer(y)
        return y

    def extra_repr(self):
        return (f'dtype:{self.dtype}, patch_size:{self.patch_size}, whitening:{self.whitening},\n'
                f'training_set:{self.training_set}, reduce_dims:{self.reduce_dims},\n'
                f'do_l2:{self.do_l2}, do_final_l2:{self.do_final_l2}, do_gmask:{self.do_gmask}\n')
