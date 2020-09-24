import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mkd_pytorch.kernel import get_grid, VonMisesKernel, COEFFS, get_kron_order


def load_fspecial_gaussian_filter(sigma):
    # Gaussian filter of size 5x5.
    rx = np.arange(-2, 3, dtype=np.float32)
    fx = np.exp(-1 * np.square(rx / (sigma * np.sqrt(2.0))))
    fx = np.expand_dims(fx, 1)
    gx = np.dot(fx, fx.T)
    gx = gx / np.sum(gx)
    return gx.astype(np.float32)


def gaussian_mask(rho, sigma=1):
    gmask = np.exp(-1 * rho**2 / sigma**2)
    return gmask


class L2Norm(nn.Module):
    def __init__(self, *, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):  # pylint: disable=W0221
        norm = torch.sqrt(torch.sum(x * x, dim=-1) + self.eps)
        x = x / norm.unsqueeze(-1)
        return x


class Gradients(nn.Module):
    def __init__(self, *, patch_size=32, do_smoothing=True, sigma=1.4, device='cuda'):
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

    def forward(self, x):  # pylint: disable=W0221
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

    def extra_repr(self):
        return f'patch_size:{self.patch_size}, pad=ReplicationPad2d'


class EmbedGradients(nn.Module):
    def __init__(self,
                 patch_size=32,
                 device='cuda',
                 relative=False):
        super().__init__()
        self.patch_size = patch_size
        self.relative = relative
        self.eps = 1e-8

        # Theta kernel for gradients.
        self.kernel = VonMisesKernel(COEFFS['theta'], patch_size=patch_size)

        # Relative gradients.
        if relative:
            grids = get_grid(patch_size)
            self.phi = torch.Tensor(grids['phi']).to(device)

    def emb_mags(self, mags):
        mags = torch.sqrt(mags + self.eps)
        return mags

    def forward(self, grads):  # pylint: disable=W0221
        mags = grads[:, :1, :, :]
        oris = grads[:, 1:, :, :]

        if self.relative:
            oris = oris - self.phi

        y = self.kernel(oris) * self.emb_mags(mags)
        return y

    def extra_repr(self):
        return f'patch_size:{self.patch_size}, relative={self.relative}, coeffs={self.kernel.coeffs}'


def spatial_kernel_embedding(dtype, patch_size):
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
    def __init__(self,
                 dtype='cart',
                 fmap_size=8,
                 in_dims=128,
                 do_gmask=True,
                 do_l2=True):
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

    def init_kron(self):
        kron = get_kron_order(self.in_dims, self.d_emb)
        idx1 = torch.Tensor(kron[:, 0]).type(torch.int64)
        idx2 = torch.Tensor(kron[:, 1]).type(torch.int64)
        emb2 = torch.index_select(self.emb, 1, idx2)
        return emb2, idx1

    # function to forward-propagate inputs through the network
    def forward(self, x):  # pylint: disable=W0221
        emb1 = torch.index_select(x, 1, self.idx1.to(x.device))
        output = emb1 * (self.emb2).to(emb1.device)
        output = output.sum(dim=(2, 3))
        if self.do_l2:
            output = self.norm(output)
        return output

    def extra_repr(self):
        return f'dtype:{self.dtype}, fmap_size:{self.fmap_size}, in_dims:{self.in_dims}, do_gmask:{self.do_gmask}, do_l2:{self.do_l2}, out_dims:{self.out_dims}'
