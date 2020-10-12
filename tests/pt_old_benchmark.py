import sys
from torch.utils.tensorboard import SummaryWriter
import torch
import mkd_pytorch as mm
from brown_phototour_revisited.benchmarking import print_results_table
import phototourism_pytorch as pp

data_dir = '/mnt/lascar/qqmukund/data/prefinale/lopa/datasets/PhotoTourism'
expt_dir = '/mnt/lascar/qqmukund/data/mkd_pytorch/expts'

if __name__ == "__main__":

    # Settings.
    max_epoch = 10
    patch_size = 32
    device = 'cuda'

    train_test_combos = {'liberty': ['notredame', 'yosemite'],
                         'notredame': ['liberty', 'yosemite'],
                         'yosemite': ['liberty', 'notredame']}

    # MKDNet.
    archs = ['orig_hardnet', 'hardnet', 'cart', 'polar']
    results = {}

    for arch in archs:
        results[arch] = {}

        # Get all 6 combinations.
        for training_set, testing_sets in train_test_combos.items():

            expt_name = f'noaug-{arch}-{training_set}-{patch_size}'
            print(f'Doing {expt_name}')

            # Initialize experiment.
            expt = pp.Experiment(expt_name, root=expt_dir)
            expt.initdirs()

            # Load model.
            model = mm.MKDNet(arch, patch_size=patch_size)
            model.device = device

            # Train model on training_set or load checkpoint.
            if expt.latest_checkpoint() == max_epoch:
                expt.load_checkpoint(model, n=max_epoch)
            else:
                trainset = pp.PhotoTourism(training_set, root=data_dir, patch_size=patch_size)
                pp.train(model,
                         trainset,
                         device=device,
                         expt=expt,
                         augment=None,
                         ntriplets=1024 * 1024,
                         batch_size=1024,
                         tensorboard_writer=SummaryWriter(str(expt.paths['logs'])))

            results[arch][training_set] = {}

            # Test model on testing_sets.
            for testing_set in testing_sets:

                # Extract descriptors on testset.
                testset = pp.PhotoTourism(testing_set, root=data_dir, patch_size=patch_size)
                patches = testset.patches.float()
                descs = pp.extract(patches,
                                   model,
                                   device=device)

                # Test descriptors on testset.
                stats = testset.eval_fpr(descs)
                stats_str = f'{training_set}:{testing_set}={stats}'
                expt.log('pt_scores', stats_str)
                results[arch][training_set][testing_set] = stats.fpr95

    print_results_table(results)
