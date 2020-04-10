#!/usr/bin/env python3

from MIDP import DataGenerator, DataLoader
import numpy as np
import argparse
import os
import yaml
import time
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--generator-config',
    required=True,
    help='generator config'
)
parser.add_argument(
    '--loader-config',
    required=True,
    help='loader config'
)
parser.add_argument(
    '--output-dir',
    default='outputs',
    help='directory to store ouptut images'
)
args = parser.parse_args()

timer = time.time()
with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)

with open(args.generator_config) as f:
    generator_config = yaml.safe_load(f)
data_gen = DataGenerator(data_loader, generator_config)


DL = data_gen.struct['DL']
PG = data_gen.struct['PG']
BG = data_gen.struct['BG']
# ensure the order
if PG.n_workers > 1:
    assert PG.ordered
assert BG.n_workers == 1
if 'AG' in data_gen.struct:
    assert data_gen.struct['AG'].n_workers == 1

os.makedirs(args.output_dir, exist_ok=True)
queue = []
scores = dict()

def dice_score(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    return 2 * np.sum(x * y) / np.sum(x + y)

# Use data list from PG since may be shuffled
progress_bar = tqdm(
    zip(PG.data_list, PG.partition),
    total=len(PG.data_list),
    dynamic_ncols=True,
    desc='[Data index]'
)

with data_gen as gen:
    for (data_idx, partition_per_data) in progress_bar:

        while len(queue) < partition_per_data:
            batch = next(gen)['label'].numpy()
            if len(queue) == 0:
                queue = batch
            else:
                queue = np.concatenate((queue, batch), axis=0)

        restored = PG.restore(data_idx, queue[:partition_per_data])
        DL.save_prediction(
            data_idx,
            restored,
            args.output_dir
        )
        queue = queue[partition_per_data:]

        scores[data_idx] = {
            roi: dice_score(
                (restored == val).astype(int),
                (DL.get_label(data_idx) == val).astype(int)
            )
            for roi, val in DL.roi_map.items()
        }

        info = '[%s] ' % data_idx
        info += ', '.join(
            '%s: %.3f' % (key, val)
            for key, val in scores[data_idx].items()
        )
        progress_bar.set_description(info)

with open(os.path.join(args.output_dir, 'score.json'), 'w') as f:
    json.dump(scores, f, indent=2)
print('Total time:', time.time()-timer)
print(scores)
