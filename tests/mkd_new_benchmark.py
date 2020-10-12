import sys
import torch
import mkd_pytorch as mm
import phototourism_pytorch as pp
from brown_phototour_revisited.benchmarking import *

data_dir = '/mnt/lascar/qqmukund/data/prefinale/lopa/datasets/PhotoTourism'
expt_dir = '/mnt/lascar/qqmukund/data/mkd_pytorch/expts'

benchmark_dir = '/mnt/lascar/qqmukund/endzone/logpolar/data/dima'
descs_out_dir = f'{benchmark_dir}/descriptors'
download_dataset_to = f'{benchmark_dir}/dataset'
results_dir = f'{benchmark_dir}/mAP'

results_dict = {}

if __name__ == "__main__":

    whitening = 'pcawt'

    # Settings.
    patch_size = int(sys.argv[1])

    dtypes = ['concat', 'cart', 'polar']

    for dtype in dtypes:

        device = 'cuda'
        training_sets = ['liberty', 'notredame', 'yosemite']
        models = {k:mm.MKD(dtype=dtype,
                           whitening=whitening,
                           training_set=k,
                           patch_size=patch_size,
                           device=device) for k in training_sets}

        expt_name = f'{dtype}-{whitening}-{patch_size}'
        results_dict[expt_name] = full_evaluation(models,
                                                  expt_name,
                                                  path_to_save_dataset=download_dataset_to,
                                                  path_to_save_descriptors=descs_out_dir,
                                                  path_to_save_mAP=results_dir,
                                                  patch_size=patch_size,
                                                  device=torch.device('cuda:0'),
                                                  distance='euclidean',
                                                  backend='pytorch-cuda')

print_results_table(results_dict)

