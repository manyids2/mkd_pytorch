import torch
import torch.nn as nn
import numpy as np


COEFFS_N1_K1 = [0.38214156, 0.48090413]
COEFFS_N2_K8 = [0.14343168, 0.268285, 0.21979234]
COEFFS_N3_K8 = [0.14343168, 0.268285, 0.21979234, 0.15838885]
COEFFS = {'xy':COEFFS_N1_K1, 'rhophi':COEFFS_N2_K8, 'theta':COEFFS_N3_K8}


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return phi, rho


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_grid(patch_size):
    x, y = [np.arange(-1 * (patch_size - 1), patch_size, 2, dtype=np.float32)] * 2
    xx, yy = np.meshgrid(x, y)
    phi, rho = cart2pol(xx, yy)
    rho = rho / np.sqrt(2 * np.power((patch_size - 1), 2))
    xx, yy = [item / (patch_size - 1) for item in [xx, yy]]
    grid = {'x':xx, 'y':yy, 'rho':rho, 'phi':phi}
    return grid


def get_kron_order(d1, d2):
    kron_order = np.zeros([d1 * d2, 2], dtype=np.int64)
    for i in range(d1):
        for j in range(d2):
            kron_order[i * d2 + j, :] = [i, j]
    return kron_order.astype(np.int64)


class VonMisesKernel(nn.Module):
    def __init__(self, coeffs, patch_size, device='cpu'):
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

    def forward(self, x):  # pylint: disable=W0221
        emb0 = self.emb0.repeat(x.size(0), 1, 1, 1)
        frange = self.frange * x
        emb1 = torch.cos(frange)
        emb2 = torch.sin(frange)
        embedding = torch.cat([emb0, emb1, emb2], dim=1)
        embedding = self.weights * embedding
        return embedding

    def extra_repr(self):
        return f'patch_size:{self.patch_size}, n:{self.n}, d:{self.d}, coeffs:{self.coeffs}'

