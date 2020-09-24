import os
import sys
import numpy as np

import torch
import mkdnet as mm

from mkd_pytorch import MKD

from mkdnet.dataset.phototourism import PhotoTourism
from mkdnet.core.extract import extract
from mkdnet.utils.files import save_dict

PT_BASE = '/mnt/lascar/qqmukund/data/prefinale/lopa/datasets/PhotoTourism/'
EXPT_BASE = '/mnt/lascar/qqmukund/data/endzone/expts'
patch_size = 64
device = 'cuda'
dims = {'cart':63, 'polar':175, 'concat':238}


def extract_test_descs(dtype, whitening, training_set, test_dataset):
    model = MKD(dtype=dtype,
                whitening=whitening,
                training_set=training_set,
                patch_size=patch_size,
                device='cuda')
    descs = extract(test_dataset.patches.float(), model, device=device)
    return descs


if __name__ == '__main__':
    whitening = sys.argv[1]
    dtype = 'concat'
    trainsets = ['liberty', 'notredame', 'yosemite']
    testsets = ['liberty', 'notredame', 'yosemite']

    scores = {}
    for testing_set in testsets:

        scores[testing_set] = {}
        for training_set in trainsets:
            print(f'train:{training_set}, test:{testing_set}')

            # Load test dataset.
            test_dataset = PhotoTourism(testing_set,
                                   root=PT_BASE,
                                   patch_size=patch_size)

            # extract descriptors.
            descs = extract_test_descs(dtype, whitening, training_set, test_dataset)

            # Get stats.
            stats = test_dataset.eval_fpr(descs)

            # Store and print.
            scores[testing_set][training_set] = {whitening: stats.fpr95 * 100}
            score = scores[testing_set][training_set]
            out_str = []
            for k,v in score.items():
                out_str.append(f'{k}: {v:2.2f}')
            print(' > '.join(out_str))

    mm.save_dict(scores, f'data/scores/mkd_pytorch_{whitening}_{dtype}.pkl')
