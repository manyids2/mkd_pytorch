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

    arch = sys.argv[1]

    # Settings.
    patch_size = 32
    device = 'cuda'
    training_sets = ['liberty', 'notredame', 'yosemite']
    models = {k:None for k in training_sets}

    # Train on all 3 datasets.
    for training_set in training_sets:

        # Initialize experiment.
        expt_name = f'{arch}-{training_set}-{patch_size}'
        expt = pp.Experiment(expt_name, root=expt_dir)
        expt.initdirs()

        model, _ = mm.load_model(arch, patch_size=patch_size, device=device)
        model.device = device  # TODO: model does not have device

        # # Train model on training_set.
        # trainset = pp.PhotoTourism(training_set, root=data_dir, patch_size=patch_size)
        # pp.train(model,
        #          trainset,
        #          device=device,
        #          expt=expt,
        #          ntriplets=1024 * 1024,
        #          batch_size=1024)

        # Load trained model.
        expt.load_checkpoint(model, n=5)
        models[training_set] = model

    expt_name = f'{arch}-cart-{patch_size}'
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
