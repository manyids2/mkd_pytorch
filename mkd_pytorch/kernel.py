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

