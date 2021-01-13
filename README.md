# Multiple Kernel Local Descriptor

**This descriptor will be part of kornia. This repository is mainly for the whitening models learned on Phototourism dataset.
Please use https://github.com/manyids2/mkd_local_descriptor as the reference implementation. **


Implementation of [Understanding and Improving Kernel Local Descriptors](https://arxiv.org/abs/1811.11147) using PyTorch.
Includes whitening models learned on PhotoTourism dataset, supervised and unsupervised versions.

## Installation
`pip install https://github.com/manyids2/mkd_pytorch/archive/v1.1.2.tar.gz`

- Tested with
  - Python 3.7.4
  - pytorch==1.5.0
  - cuda 10.1 (Cuda compilation tools, release 10.1, V10.1.243)
  - Nvidia Tesla P100

## Usage

*NOTE* Uses the 'pcawt' version (unsupervised whitening with attenuation) as default as it generalizes better. However, the 'lw' (learned whitening) performs better on PhotoTourism benchmark.

```python
import torch
import mkd_local_descriptor as mm

# Use model_file trained on PhotoTourism for 64x64 patch_size.

mkd = mm.MKD(dtype='concat',             # 'concat', 'polar', 'cart'
             patch_size=64,
             whitening='pcawt',          # None, 'lw', 'pca', 'pcaws', 'pcawt'
             training_set='liberty',     # 'liberty', 'notredame', 'yosemite'
             reduce_dims=128,
             do_l2=True,
             do_final_l2=True,
             do_gmask=True,
             device='cpu')

patches = torch.rand(12, 1, 64, 64)
descs = mkd(patches)
print(f'descs: {descs.shape}')
```

## Performance

Evaluated on the [brown_phototour_revisited benchmark](https://github.com/ducha-aiki/brown_phototour_revisited).

```
@misc{BrownRevisited2020,
  title={UBC PhotoTour Revisied},
  author={Mishkin, Dmytro},
  year={2020},
  url = {https://github.com/ducha-aiki/brown_phototour_revisited}
}
```


- patch_size = 64
```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-64     56.58              47.68               48.20
  MKD-concat-None-64  57.48  57.48        49.49  49.49        48.29  48.29
  MKD-concat-lw-64    73.36  73.15        61.90  59.95        61.94  60.35
  MKD-concat-pca-64   65.58  65.36        55.53  54.60        55.03  54.46
  MKD-concat-pcaws-64 67.74  67.16        57.14  55.80        56.13  55.63
  MKD-concat-pcawt-64 69.44  68.96        58.19  56.59        57.41  56.55
  ------------------------------------------------------------------------------
```

- patch_size = 32

```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-32        58.24              49.07               49.65
  MKD-concat-None-32    57.86  57.86        49.93  49.93        48.78  48.78
  MKD-concat-lw-32      72.29  71.98        60.90  58.81        60.71  59.13
  MKD-concat-pca-32     64.46  64.32        54.82  53.87        54.13  53.62
  MKD-concat-pcaws-32   66.59  66.05        56.43  55.21        55.22  54.78
  MKD-concat-pcawt-32   68.05  67.61        57.22  55.76        56.27  55.50
  ------------------------------------------------------------------------------
```

## Extraction times

Extraction times for 1024 batches on GPU (Nvidia Tesla P100).

|patch_size|batch_size| time(seconds)| ms per patch
|:--------:|:--------:|:------------:|:-----------:
|    64    | 64       |  8.96        | 13.67e-2
|    64    | 128      | 10.50        |  8.01e-2
|    64    | 256      | 19.15        |  7.30e-2
|    64    | 512      | 36.98        |  7.05e-2
|    64    | 1024     | 72.81        |  6.94e-2

|patch_size|batch_size| time(seconds)| ms per patch
|:--------:|:--------:|:------------:|:---------:
|    32    | 64       |  5.93        |  9.04e-2
|    32    | 128      |  4.21        |  3.21e-2
|    32    | 256      |  5.67        |  2.16e-2
|    32    | 512      |  9.12        |  1.73e-2
|    32    | 1024     | 17.26        |  1.64e-2

## Bibliography

  Please cite :
```
  @article{mukundan2017multiple,
    title={Multiple-kernel local-patch descriptor},
    author={Mukundan, Arun and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
    journal={arXiv preprint arXiv:1707.07825},
    year={2017}
  }

  @article{mukundan2018understanding,
    title={Understanding and improving kernel local descriptors},
    author={Mukundan, Arun and Tolias, Giorgos and Bursuc, Andrei and J{\'e}gou, Herv{\'e} and Chum, Ond{\v{r}}ej},
    journal={International Journal of Computer Vision},
    year={2018},
  }
```


## Testing all variants, patch_sizes

```
  -----------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  -----------------------------------------------------------------------------------
  trained on                 liberty notredame  liberty yosemite  notredame yosemite
  tested  on                     yosemite           notredame            liberty
  -----------------------------------------------------------------------------------
  mkd_pytorch-cart-None-32        40.27              37.64               38.46
  mkd_pytorch-cart-lw-32      51.13  51.04        45.28  44.68        45.43  45.14
  mkd_pytorch-cart-pca-32     44.72  44.35        40.53  40.22        41.50  41.28
  mkd_pytorch-cart-pcaws-32   48.95  48.72        43.80  43.35        43.94  43.93
  mkd_pytorch-cart-pcawt-32   49.38  49.02        43.90  43.33        43.92  43.85
  mkd_pytorch-polar-None-32       61.49              52.98               51.56
  mkd_pytorch-polar-lw-32     73.20  73.12        62.67  61.34        63.07  62.00
  mkd_pytorch-polar-pca-32    66.25  65.98        56.59  55.57        55.54  55.03
  mkd_pytorch-polar-pcws-32   67.69  67.20        57.94  56.82        56.37  56.00
  mkd_pytorch-polar-pcwt-32   69.53  69.06        58.81  57.39        57.65  56.91
  mkd_pytorch-concat-None-32      57.86              49.93               48.78
  mkd_pytorch-concat-lw-32    72.29  71.98        60.90  58.81        60.71  59.13
  mkd_pytorch-concat-pca-32   64.46  64.32        54.82  53.87        54.13  53.62
  mkd_pytorch-concat-pcws-32  66.59  66.05        56.43  55.21        55.22  54.78
  mkd_pytorch-concat-pcwt-32  68.05  67.61        57.22  55.76        56.27  55.50

  mkd_pytorch-cart-None-64        38.93              36.71               37.62
  mkd_pytorch-cart-lw-64      51.61  51.44        45.67  44.97        45.74  45.40
  mkd_pytorch-cart-pca-64     44.61  44.26        40.46  40.12        41.50  41.31
  mkd_pytorch-cart-pcws-64    49.49  49.15        44.12  43.56        44.34  44.28
  mkd_pytorch-cart-pcwt-64    49.60  49.20        43.95  43.34        44.07  43.96
  mkd_pytorch-polar-None-64      61.57              52.93               51.45
  mkd_pytorch-polar-lw-64     73.95  73.97        63.59  62.13        64.05  62.87
  mkd_pytorch-polar-pca-64    67.37  67.09        57.45  56.46        56.48  56.01
  mkd_pytorch-polar-pcws-64   68.80  68.28        58.77  57.54        57.30  56.84
  mkd_pytorch-polar-pcwt-64   70.91  70.39        59.97  58.39        58.86  58.03
  mkd_pytorch-concat-None-64     57.48              49.49               48.29
  mkd_pytorch-concat-lw-64    73.36  73.15        61.90  59.95        61.94  60.35
  mkd_pytorch-concat-pca-64   65.58  65.36        55.53  54.60        55.03  54.46
  mkd_pytorch-concat-pcws-64  67.74  67.16        57.14  55.80        56.13  55.63
  mkd_pytorch-concat-pcwt-64  69.44  68.96        58.19  56.59        57.41  56.55
  ----------------------------------------------------------------------------------
```

