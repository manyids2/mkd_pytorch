import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def input_norm(x, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize patches for 0.0 mean and 1.0 std

    Args:
        x : (Tensor) Batch of patches to normalize
        eps : (float) Epsilon to account for division by 0 in std (1e-8 is default)

    Returns:
        Tensor: Normalized patches
    Shape:
        - Input: (B, 1, patch_size, patch_size)
        - Output: (B, 1, patch_size, patch_size)
    """
    flat = x.contiguous().view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + eps
    x = (x - mp.view(-1, 1, 1, 1)) / sp.view(-1, 1, 1, 1)
    return x


def weights_init(m: nn.Module) -> None:
    """
    Initialization of model parameters
    If either nn.Conv2d or nn.Linear, use orthogonal_ initialization

    Args:
    m : (nn.Module) pytorch module to initialize weights for
    """
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=0.6)


class ReluOpt(nn.Module):
    """Helper layer for optional ReLU

    Args:
        do_relu :  (bool) Do relu or bypass (True is default)
    """

    def __init__(self, *, do_relu:bool = True) -> None:
        super().__init__()
        self.do_relu = do_relu
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x) if self.do_relu else x

    def extra_repr(self) -> str:
        return f'do_relu={self.do_relu}'


class Reshape(nn.Module):
    """Helper layer to reshape a Tensor

    Args:
    shape : (list) shape to reshape to
    """

    def __init__(self, *args) -> None:
        super().__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)

    def extra_repr(self) -> str:
        return f'reshape:{self.shape}'


class Conv_BN_Relu(nn.Module):
    """Helper layer for Convolution, Batch Normalization, then ReLU

    Args:
        idims :  (int) Dimensionality of input features
        odims :  (int) Dimensionality of output features
        kernel_size :  (int) kernel size for convolutional layer (3 is default)
        stride :  (int) stride for convolutional layer (1 is default)
        bias :  (bool) Use bias for convolutional layer (True is default)
        affine :  (bool) Use affine for batchnorm (True is default)
        do_relu :  (bool) Do relu or bypass (True is default)
    """

    def __init__(self,
                 idims: int,
                 odims: int,
                 *,
                 kernel_size: int = 3,
                 padding: int = 0,
                 stride: int = 1,
                 bias: bool = False,
                 affine: bool = False,
                 do_relu: bool = True) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(idims,
                      odims,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=bias),
            nn.BatchNorm2d(odims, affine=affine),
            ReluOpt(do_relu=do_relu),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def orig_FCN() -> nn.Sequential:
    """Original HardNet implementation. """
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32, affine=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32, affine=False),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64, affine=False),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64, affine=False),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128, affine=False),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128, affine=False),
        nn.ReLU(),
        nn.Dropout(0.1),
    )


def FCN() -> nn.Sequential:
    """Fully convolutional network, based on HardNet. """
    return nn.Sequential(
        Conv_BN_Relu(1, 32, kernel_size=3, padding=1),
        Conv_BN_Relu(32, 32, kernel_size=3, padding=1),
        Conv_BN_Relu(32, 64, kernel_size=3, padding=1, stride=2),
        Conv_BN_Relu(64, 64, kernel_size=3, padding=1),
        Conv_BN_Relu(64, 128, kernel_size=3, padding=1, stride=2),
        Conv_BN_Relu(128, 128, kernel_size=3, padding=1),
        nn.Dropout(p=0.1),
    )


def FC() -> nn.Sequential:
    """Fully connected layer, based on HardNet. """
    return nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
    )


class DimensionalityReduction(nn.Module):
    def __init__(self, idims: int, odims: int, *, with_bias: bool = False):
        super().__init__()
        self.idims = idims
        self.odims = odims
        self.with_bias = with_bias

        self.weight = Parameter(torch.Tensor(
            np.ones([idims, odims], dtype=np.float32)),
                                requires_grad=True)
        nn.init.orthogonal_(self.weight.data, gain=0.6)

        if self.with_bias:
            self.bias = Parameter(torch.Tensor(np.ones([idims], dtype=np.float32)),
                                  requires_grad=True)
            nn.init.constant_(self.bias.data, 0.00)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_bias:
            x = x - self.bias
        output = x @ self.weight
        return output

    def extra_repr(self) -> str:
        return (f'W : {self.weight.shape}, bias: {self.with_bias}')


def DimensionalityReductionBN(idims: int, odims: int) -> nn.Sequential:
    """Helper for DimensionalityReduction with batch norm. """
    return nn.Sequential(
        DimensionalityReduction(idims=idims, odims=odims, with_bias=True),
        Reshape(-1, odims, 1, 1), nn.BatchNorm2d(odims, affine=False),
        Reshape(-1, odims))

