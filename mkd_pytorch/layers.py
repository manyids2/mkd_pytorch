from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import COEFFS, get_grid, get_kron_order


def load_fspecial_gaussian_filter(sigma: float) -> np.ndarray:
    """Gaussian filter of size 5x5. Matches Matlab implementation of kde. """
    rx = np.arange(-2, 3, dtype=np.float32)
    fx = np.exp(-1 * np.square(rx / (sigma * np.sqrt(2.0))))
    fx = np.expand_dims(fx, 1)
    gx = np.dot(fx, fx.T)
    gx = gx / np.sum(gx)
    return gx.astype(np.float32)


def gaussian_mask(rho: np.ndarray, sigma: float = 1.0) -> np.nparray:
    """Compute gaussian mask given distance from centre (rho). """
    gmask = np.exp(-1 * rho**2 / sigma**2)
    return gmask


class L2Norm(nn.Module):
    """l2-normalization as layer. """

    def __init__(self, *, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        norm = torch.sqrt(torch.sum(x * x, dim=-1) + self.eps)
        x = x / norm.unsqueeze(-1)
        return x


class VonMisesKernel(nn.Module):
    """
    Module, which computes parameters of Von Mises kernel given coefficients,
    and embeds given patches.
    Args:
        coeffs: (list) List of coefficients
              Some examples are hardcoded in COEFFS
        patch_size: (int) Input patch size in pixels (32 is default)
        device: (str) Torch device to use ('cpu' is default)
    Returns:
        Tensor: Von Mises embedding of given parametrization
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, out_dim, patch_size, patch_size)
    Examples::
        >>> mags = torch.rand(23, 1, 32, 32)
        >>> vm = mkd_pytorch.VonMisesKernel(coeffs=mkd_pytorch.COEFFS['xy'],
                                            patch_size=32,
                                            device='cpu')
        >>> emb = vm(mags) # 23x3x32x32
    """

    def __init__(self,
                 coeffs: Union[list, tuple, np.ndarray],
                 patch_size: int,
                 device: str = 'cpu'):
        super().__init__()

        self.coeffs = np.array(coeffs, dtype=np.float32)
        self.patch_size = patch_size

        n = self.coeffs.shape[0] - 1
        self.n = n
        self.d = 2 * n + 1

        weights = np.zeros([2 * n + 1], dtype=np.float32)
        weights[:n + 1] = np.sqrt(self.coeffs)
        weights[n + 1:] = np.sqrt(self.coeffs[1:])
        weights = weights.reshape(-1, 1, 1).astype(np.float32)  # pylint: disable=E1121
        weights = torch.Tensor(weights).to(device)

        frange = np.arange(n) + 1
        frange = frange.reshape(-1, 1, 1).astype(np.float32)
        frange = torch.Tensor(frange).to(device)

        self.emb0 = torch.ones([1, 1, patch_size, patch_size]).to(device)
        self.frange = frange
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        emb0 = self.emb0.repeat(x.size(0), 1, 1, 1)
        frange = self.frange * x
        emb1 = torch.cos(frange)
        emb2 = torch.sin(frange)
        embedding = torch.cat([emb0, emb1, emb2], dim=1)
        embedding = self.weights * embedding
        return embedding

    def extra_repr(self) -> str:
        return f'patch_size:{self.patch_size}, n:{self.n}, d:{self.d}, coeffs:{self.coeffs}'


class Gradients(nn.Module):
    """
    Module, which computes gradients of given patches, stacked as [magnitudes, orientations].
    Args:
        patch_size: (int) Input patch size in pixels (32 is default)
        do_smoothing: (bool) Smooth patches before computing gradients (True is default)
        sigma: (float) Sigma of gaussian filter for smoothing a 64x64 patch (1.4 is default)
        device: (str) Torch device to use ('cpu' is default)
    Returns:
        Tensor: gradients of given patches
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, 2, patch_size, patch_size)
    Examples::
        >>> patches = torch.rand(23, 1, 32, 32)
        >>> grads = mkd_pytorch.Gradients(patch_size=32,
                                          do_smoothing=True,
                                          sigma=1.4,
                                          device='cpu')
        >>> g = grads(patches) # 23x2x32x32
    """

    def __init__(self,
                 patch_size: int = 32,
                 do_smoothing: bool = True,
                 sigma: float = 1.4,
                 device: str = 'cpu') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.do_smoothing = do_smoothing
        self.sigma = sigma / 64.0  # 1.4 at patch_size = 64
        self.eps = 1e-8

        # Basic filter - 1, 0, -1.
        delta = np.array([1, 0, -1])
        xf = delta.reshape([1, 3])
        yf = delta.reshape([3, 1])
        xf, yf = [
            f_[np.newaxis, np.newaxis, :, :].astype(np.float32) for f_ in [xf, yf]
        ]
        self.xf = torch.Tensor(xf).to(device)
        self.yf = torch.Tensor(yf).to(device)
        self.gp = nn.ReplicationPad2d((2, 2, 2, 2))
        self.xp = nn.ReplicationPad2d((1, 1, 0, 0))
        self.yp = nn.ReplicationPad2d((0, 0, 1, 1))
        self.bias = torch.zeros([1], dtype=torch.float32).to(device)

        # Smoothing filter taken from Matlab function fspecial_gaussian.
        if do_smoothing:
            gaussian = load_fspecial_gaussian_filter(sigma)
            gaussian = gaussian[np.newaxis, np.newaxis, :, :].astype(np.float32)
            self.gaussian = torch.Tensor(gaussian).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        # Gaussian smoothing.
        if self.do_smoothing:
            x = self.gp(x)
            x = F.conv2d(x, self.gaussian, self.bias, 1, 0)

        # x and y gradients.
        gx = F.conv2d(self.xp(x), self.xf, self.bias, 1, 0)
        gy = F.conv2d(self.yp(x), self.yf, self.bias, 1, 0)

        # Magnitudes and orientations.
        mags = torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2) + self.eps)
        oris = torch.atan2(gy, gx)

        # Concatenate and return.
        y = torch.cat([mags, oris], dim=1)
        return y

    def extra_repr(self) -> str:
        return f'patch_size:{self.patch_size}, pad=ReplicationPad2d'


class EmbedGradients(nn.Module):
    """
    Module, orientation embedding weighed by sqrt of magnitudes of given patches.
    Args:
        patch_size: (int) Input patch size in pixels (32 is default)
        relative: (bool) Whether to compute absolute or relative gradients (False is default)
        device: (str) Torch device to use ('cpu' is default)
    Returns:
        Tensor: Orientation embedding weighed by sqrt of magnitudes of given patches
    Shape:
        - Input: (B, 2, patch_size, patch_size)
        - Output: (B, 7, patch_size, patch_size)
    Examples::
        >>> grads = torch.rand(23, 1, 32, 32)
        >>> emb_grads = mkd_pytorch.EmbedGradients(patch_size=32,
                                               relative=True,
                                               device='cpu')
        >>> emb = emb_grads(patches) # 23x7x32x32
    """

    def __init__(self,
                 patch_size: int = 32,
                 relative: bool = False,
                 device: str = 'cpu') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.relative = relative
        self.eps = 1e-8

        # Theta kernel for gradients.
        self.kernel = VonMisesKernel(COEFFS['theta'], patch_size=patch_size, device=device)

        # Relative gradients.
        if relative:
            grids = get_grid(patch_size)
            self.phi = torch.Tensor(grids['phi']).to(device)

    def emb_mags(self, mags: torch.Tensor) -> torch.Tensor:
        """Embed square roots of magnitudes with eps for numerical reasons. """
        mags = torch.sqrt(mags + self.eps)
        return mags

    def forward(self, grads: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        mags = grads[:, :1, :, :]
        oris = grads[:, 1:, :, :]

        if self.relative:
            oris = oris - self.phi

        y = self.kernel(oris) * self.emb_mags(mags)
        return y

    def extra_repr(self) -> str:
        return f'patch_size:{self.patch_size}, relative={self.relative}, coeffs={self.kernel.coeffs}'


def spatial_kernel_embedding(dtype, patch_size: int) -> torch.Tensor:
    """Compute embeddings for hardcoded cartesian and polar parametrizations. """
    factors = {"phi": 1.0, "rho": np.pi, "x": np.pi / 2, "y": np.pi / 2}
    if dtype == 'cart':
        coeffs_ = 'xy'
        params_ = ['x', 'y']
    elif dtype == 'polar':
        coeffs_ = 'rhophi'
        params_ = ['phi', 'rho']

    grids = get_grid(patch_size)
    grids_normed = {k:v * factors[k] for k,v in grids.items()}
    grids_normed = {k:v[np.newaxis, np.newaxis, :, :] for k,v in grids_normed.items()}
    grids_normed = {k:torch.from_numpy(v.astype(np.float32)) for k,v in grids_normed.items()}

    vm_a = VonMisesKernel(COEFFS[coeffs_], patch_size=patch_size)
    vm_b = VonMisesKernel(COEFFS[coeffs_], patch_size=patch_size)

    emb_a = vm_a(grids_normed[params_[0]]).squeeze()
    emb_b = vm_b(grids_normed[params_[1]]).squeeze()

    kron_order = torch.from_numpy(get_kron_order(vm_a.d, vm_b.d).astype(np.int64))
    spatial_kernel = emb_a.index_select(0, kron_order[:,0]) * emb_b.index_select(0, kron_order[:,1])
    return spatial_kernel


class ExplicitSpacialEncoding(nn.Module):
    """
    Module, computes sum and l2-norm of explicit cartesian or polar spatial embedding given a feature map.
    Args:
        dtype: (str) Parametrization of kernel. 'polar', 'cart' ('polar' is default)
        fmap_size: (int) Input feature map size in pixels (32 is default)
        in_dims: (int) Dimensionality of input feature map (7 is default)
        do_gmask: (bool) Apply gaussian mask to emphasize center (True is default)
        do_l2: (bool) Apply l2-normalization to output vector (True is default)
    Returns:
        Tensor: Orientation embedding weighed by sqrt of magnitudes of given patches
    Shape:
        - Input: (B, in_dims, fmap_size, fmap_size)
        - Output: (B, out_dims, fmap_size, fmap_size)
    Examples::
        >>> emb_ori = torch.rand(23, 7, 32, 32)
        >>> polar_emb = mkd_pytorch.ExplicitSpacialEncoding(dtype='polar',
                                                            fmap_size=32,
                                                            in_dims=7,
                                                            do_gmask=True,
                                                            do_l2=True)
        >>> desc = polar_emb(emb_ori) # 23x175x32x32
    """

    def __init__(self,
                 dtype: str = 'polar',
                 fmap_size: int = 32,
                 in_dims: int = 7,
                 do_gmask: bool = True,
                 do_l2: bool = True) -> None:
        super().__init__()

        self.dtype = dtype
        self.fmap_size = fmap_size
        self.in_dims = in_dims
        self.do_gmask = do_gmask
        self.do_l2 = do_l2
        self.grid = get_grid(fmap_size)
        self.gmask = None

        if self.dtype == 'cart':
            emb = spatial_kernel_embedding('cart', self.fmap_size)
        elif self.dtype == 'polar':
            emb = spatial_kernel_embedding('polar', self.fmap_size)
        else:
            raise NotImplementedError(f'{self.dtype} is not implemented.')

        if self.do_gmask:
            rho = self.grid['rho'] / self.grid['rho'].max()
            self.gmask = gaussian_mask(rho, sigma=1.0)
            emb = emb * self.gmask

        self.emb = emb.unsqueeze(0)
        self.d_emb = self.emb.shape[1]
        self.out_dims = self.in_dims * self.d_emb
        self.odims = self.out_dims

        self.emb2, self.idx1 = self.init_kron()
        self.norm = L2Norm()

    def init_kron(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize helper variables to calculate kronecker. """
        kron = get_kron_order(self.in_dims, self.d_emb)
        idx1 = torch.Tensor(kron[:, 0]).type(torch.int64)
        idx2 = torch.Tensor(kron[:, 1]).type(torch.int64)
        emb2 = torch.index_select(self.emb, 1, idx2)
        return emb2, idx1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        emb1 = torch.index_select(x, 1, self.idx1.to(x.device))
        output = emb1 * (self.emb2).to(emb1.device)
        output = output.sum(dim=(2, 3))
        if self.do_l2:
            output = self.norm(output)
        return output

    def extra_repr(self) -> str:
        return (f'dtype:{self.dtype}, fmap_size:{self.fmap_size} \n',
                f'in_dims:{self.in_dims}, out_dims:{self.out_dims} \n, '
                f'do_gmask:{self.do_gmask}, do_l2:{self.do_l2}')


class Whitening(nn.Module):
    """
    Module, performs supervised or unsupervised whitening as described in
    [Understanding and Improving Kernel Local Descriptors](https://arxiv.org/abs/1811.11147) .
    Args:
        xform: (str) Variant of whitening to use. None, 'lw', 'pca', 'pcaws', 'pcawt'
        whitening_model: (dict) Dictionary with keys 'mean', 'eigvecs', 'eigvals'
              holding appropriate numpy arrays
        reduce_dims: (int) Dimensionality reduction (128 is default)
        keval: (int) Shrinkage parameter (40 is default)
        t: (float) Attenuation parameter (0.7 is default)
        device: (str) Torch device to use ('cpu' is default)
    Returns:
        Tensor: Orientation embedding weighed by sqrt of magnitudes of given patches.
    Shape:
        - Input: (B, in_dims, fmap_size, fmap_size)
        - Output: (B, out_dims, fmap_size, fmap_size)
    Examples::
        >>> descs = torch.rand(23, 238)
        >>> whitening = mkd_pytorch.Whitening(xform='pcawt',
                                              whitening_model,
                                              reduce_dims=128,
                                              keval=40,
                                              t=0.7,
                                              device='cpu')
        >>> wdescs = whitening(descs) # 23x128
    """

    def __init__(self,
                 xform: str,
                 whitening_model: dict,
                 reduce_dims: int = 128,
                 keval: int = 40,
                 t: float = 0.7,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.xform = xform
        self.reduce_dims = reduce_dims
        self.keval = keval
        self.t = t
        self.norm = L2Norm()

        self.whitening_model = {k:torch.from_numpy(v.astype(np.float32)) for k,v in whitening_model.items()}
        self.whitening_model = {k:v.to(device) for k,v in self.whitening_model.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0221
        # Center the data.
        x = x - self.whitening_model["mean"]

        # Dimensionality reduction.
        data_dim = x.shape[1]
        evecs = self.whitening_model["eigvecs"][:, :min(self.reduce_dims, data_dim)]
        evals = self.whitening_model["eigvals"][:min(self.reduce_dims, data_dim)]

        # Transform.
        pval = 1.0
        if self.xform == 'pca':
            x = x @ evecs
            pval = 0.5
        elif self.xform == 'whiten':
            evecs = evecs @ torch.diag(torch.pow(evals, -0.5))
            x = x @ evecs
        elif self.xform == 'lw':
            x = x @ evecs
        elif self.xform == 'pcaws':
            alpha = evals[self.keval]
            evals = ((1 - alpha) * evals) + alpha
            evecs = evecs @ torch.diag(torch.pow(evals, -0.5))
            x = x @ evecs
        elif self.xform == 'pcawt':
            m = -0.5 * self.t
            evecs = evecs @ torch.diag(torch.pow(evals, m))
            x = x @ evecs
        else:
            raise KeyError('Unknown transform - %s' % self.xform)

        # powerlaw.
        x = torch.sign(x) * torch.pow(torch.abs(x), pval)

        # l2norm
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f'xform:{self.xform}, reduce_dims:{self.reduce_dims}'
