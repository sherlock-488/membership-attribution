import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import numpy as np

SAVE_WEIGHTS_PATH = '../'
SAVE_PATH = './'

def load_train_data(dataname, training_mode, training_stage):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')
    else:
        raise ValueError('Invalid dataset name')

    data = dataset.shuffle(seed=2023).select_columns('text')
    if training_mode not in ['target', 'shadow']:
        raise ValueError('Invalid training mode')
    data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
    D_1 = data.select(range(0, 2500))
    D_2 = data.select(range(2500, 5000))
    D_3 = data.select(range(5000, 7500))
    if training_stage not in ['first', 'second']:
        raise ValueError('Invalid training stage')
    dataset = concatenate_datasets([D_2, D_3]) if training_stage == 'first' else concatenate_datasets([D_1, D_2]) 
    return dataset

def load_eval_data(dataname, training_mode):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')

    data = dataset.shuffle(seed=2023).select_columns('text')
    data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
    return data


class shadowDataset(Dataset):
    def __init__(self, filepath, target_mode):
        if target_mode == 'pt_pt':
            num_per_class = 1000
        else:
            num_per_class = 2500
        self.preds = torch.load(filepath)
        self.num_per_class = num_per_class
        self.num_classes = len(self.preds)//self.num_per_class
        
    def get_dim(self):
        return self.preds[0].shape[1]
    
    def __len__(self):
        return len(self.preds)
    
    def __getitem__(self, idx):
        return self.preds[idx], idx//self.num_per_class