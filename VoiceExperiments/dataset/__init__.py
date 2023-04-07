import torchaudio
from torch.utils.data import ConcatDataset, random_split

from VoiceExperiments.dataset.vctk import VCTKDataset

def get_datasets(dataset_configs):
    # TODO: add support for weighted sampling
    # TODO: add support for selecting batch keys from config

    train_datasets, val_datasets = [], []

    # parase each dataset config and split if needed
    for config in dataset_configs:
        full_dataset = _get_dataset(config)

        if config['val_split'] <= 0:
            train_datasets.append(full_dataset)
        elif config['val_split'] >= 1:
            val_datasets.append(full_dataset)
        else:
            train_dataset, val_dataset = random_split(full_dataset, [1-config['val_split'], config['val_split']])
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
    
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)

        

def _get_dataset(config):
    name = config['name'].lower()

    if name == 'vctk':
        return VCTKDataset(**config['kwargs'])

    raise NotImplementedError(f'Dataset {name} not found')