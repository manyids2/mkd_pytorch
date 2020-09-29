import torch
import torch.nn as nn

from .deeplayers import weights_init, input_norm, orig_FCN, FC, FCN, DimensionalityReductionBN
from .layers import L2Norm, Whitening, ExplicitSpacialEncoding


class MKDNet(nn.Module):
    def __init__(self,
                 arch: str,
                 patch_size: int = 32) -> None:
        super().__init__()

        self.arch = arch
        self.patch_size = patch_size
        self.fmap_size = patch_size // 4

        self.l2norm = L2Norm()

        # FCN + FC.
        # TODO: add polarlog, polarlinear, polarnaive.
        if self.arch == 'orig_hardnet':
            self.fcn = orig_FCN()
            self.fc = FC()
        elif self.arch == 'hardnet':
            self.fcn = FCN()
            self.fc = FC()
        elif self.arch == 'cart':
            self.fcn = FCN()
            self.encoding = ExplicitSpacialEncoding(dtype='cart', fmap_size=self.fmap_size)
            self.fc = nn.Sequential(self.encoding, DimensionalityReductionBN(self.encoding.out_dims, 128))
        elif self.arch == 'polar':
            self.fcn = FCN()
            self.encoding = ExplicitSpacialEncoding(dtype='polar', fmap_size=self.fmap_size)
            self.fc = nn.Sequential(self.encoding, DimensionalityReductionBN(self.encoding.out_dims, 128))
        else:
            raise NotImplementedError(f'{self.arch} not implemented.')

        # Common architecture.
        self.features = nn.Sequential(self.fcn, self.fc)

        # initialize weights
        self.features.apply(weights_init)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x_features = self.features(input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        x = self.l2norm(x)
        return x
