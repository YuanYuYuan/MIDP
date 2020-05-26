#!/usr/bin/env python3

from MIDP import DataGenerator, DataLoader, Reverter
import argparse
import os
import yaml
import time
from tqdm import tqdm

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


os.makedirs(args.output_dir, exist_ok=True)


DL = data_gen.struct['DL']
batch_list = list({'prediction': data['label']} for data in data_gen)
reverter = Reverter(data_gen)
progress_bar = tqdm(
    reverter.on_batches(batch_list),
    total=len(reverter.data_list),
    dynamic_ncols=True,
    desc='[Data index]'
)

scores = dict()
for result in progress_bar:
    data_idx = result['idx']
    DL.save_prediction(
        data_idx,
        result['prediction'],
        args.output_dir
    )
    scores[data_idx] = DL.evaluate(data_idx, result['prediction'])

    info = '[%s] ' % data_idx
    info += ', '.join(
        '%s: %.3f' % (key, val)
        for key, val in scores[data_idx].items()
    )
    progress_bar.set_description(info)

print('Total time:', time.time()-timer)
print(scores)
