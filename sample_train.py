#!/usr/bin/env python3
from MIDP import DataGenerator, DataLoader
import argparse
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='config including data generator'
)
args = parser.parse_args()


# load config
with open(args.config) as f:
    config = yaml.safe_load(f)
generator_config = config['generator']
with open(config['data']) as f:
    data_config = yaml.safe_load(f)
data_list = data_config['list']
loader_config = data_config['loader']

# data pipeline
data_gen = dict()
for stage in ['train', 'valid']:
    data_loader = DataLoader(*loader_config)
    if data_list[stage] is not None:
        data_loader.set_data_list(data_list[stage])
    data_gen[stage] = DataGenerator(data_loader, generator_config[stage])


print('Start training.')
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    print('========== Epoch %d ==========' % (epoch+1))

    print('Training...')
    for data in tqdm(data_gen['train']):
        image = data['image']
        label = data['label']

        '''
        Feed data into model and do back propagation
        '''

    print('Validating...')
    for data in tqdm(data_gen['valid']):
        image = data['image']
        label = data['label']

        '''
        Feed data into model and check validation performance
        '''
print('Finished training.')
