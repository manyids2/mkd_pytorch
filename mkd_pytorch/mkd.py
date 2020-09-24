import os
import numpy as np
import torch
import torch.nn as nn

from .layers import Gradients, EmbedGradients, ExplicitSpacialEncoding, L2Norm, Whitening


class MKD(nn.Module):
    def __init__(self, dtype='concat',
                 patch_size=32,
                 whitening='pcawt',
                 training_set='liberty',
                 reduce_dims=128,
                 do_l2=True,
                 do_final_l2=True,
                 do_gmask=True,
                 device='cpu'):
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

        # Use the correct model_file.
        this_dir, _ = os.path.split(__file__)
        self.model_file = os.path.join(this_dir, f'mkd-{dtype}-64.pkl')

        self.grads = Gradients(patch_size=patch_size,
                               do_smoothing=True,
                               sigma=1.4 * (patch_size / 64.0),
                               device=device)

        if self.whitening is not None:
            whitening_models = np.load(self.model_file, allow_pickle=True)
            algo = 'lw' if self.whitening == 'lw' else 'pca'
            whitening_model = whitening_models[self.training_set][algo]
            self.whitening_layer = Whitening(self.whitening,
                                             whitening_model,
                                             reduce_dims=self.reduce_dims,
                                             device=self.device)

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

        self.norm = L2Norm()
        if dtype == 'concat':
            self.odims = polar_emb.odims + cart_emb.odims
        elif dtype == 'cart':
            self.odims = cart_emb.odims
        elif dtype == 'polar':
            self.odims = polar_emb.odims

        self.out_dim = self.odims
        if self.whitening is not None:
            self.out_dim = self.reduce_dims
            self.odims = self.reduce_dims

    def forward(self, patches):  # pylint: disable=W0221
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
