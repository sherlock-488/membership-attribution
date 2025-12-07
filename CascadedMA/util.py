import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    OpenAIConfig,
)
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.utils import fix_seed
from mimir.models import LanguageModel, ReferenceModel
from mimir.attacks.all_attacks import AllAttacks, Attack
from mimir.attacks.neighborhood import T5Model, BertModel, NeighborhoodAttack
from mimir.attacks.utils import get_attacker
SAVE_PATH = '../'

def load_data(dataname, training_mode):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')

    data = dataset.shuffle(seed=2023).select_c
    data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
    D_1 = data.select(range(0, 2500))
    D_2 = data.select(range(5000, 10000))
    return concatenate_datasets([D_1, D_2])

def load_data_delta(dataname, training_mode):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')

    data = dataset.shuffle(seed=2023).select_columns('text')

    data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
    return data

def load_multi_data(dataname, training_mode, model_index, stage):
    dataset = load_dataset(dataname, split='train')
    data = dataset.shuffle(seed=2023).select_columns('text')
    pretrained_data = load_dataset("LazarusNLP/mini_pile_cc", split='train').shuffle(seed=2023).select_columns('text')
    if training_mode == 'target':
        D_1 = data.select(range(0, 2500))
        D_2 = pretrained_data.select(range(0, 2500))
        D_3 = pretrained_data.select(range(2500, 5000))
        D_4 = data.select(range(2500, 5000))
        D_5 = data.select(range(10000, 12500))
        D_6 = pretrained_data.select(range(10000, 12500))
        D_7 = pretrained_data.select(range(12500, 15000))
        D_8 = data.select(range(12500, 15000))
    elif training_mode == 'shadow':
        D_1 = data.select(range(5000, 7500))
        D_2 = pretrained_data.select(range(5000, 7500))
        D_3 = pretrained_data.select(range(7500, 10000))
        D_4 = data.select(range(7500, 10000))
        D_5 = data.select(range(15000, 17500))
        D_6 = pretrained_data.select(range(15000, 17500))
        D_7 = pretrained_data.select(range(17500, 20000))
        D_8 = data.select(range(17500, 20000))
    else:
        raise ValueError("Not for target or shadow.")
    
    if stage == 'first':
        dataset = concatenate_datasets([D_1, D_2, D_5, D_6]) 
    elif stage == 'second':
        dataset = concatenate_datasets([D_5, D_6, D_7, D_8]) 
    elif stage == 'no_overlap':
        dataset = concatenate_datasets([D_1, D_3, D_4, D_8])
    else:
        dataset = concatenate_datasets([D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8])
    return dataset

def get_attackers(
    target_model,
    ref_models,
    config: ExperimentConfig,
):
    # Look at all attacks, and attacks that we have implemented
    attacks = config.blackbox_attacks
    implemented_blackbox_attacks = [a.value for a in AllAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            continue
        runnable_attacks.append(a)
    attacks = runnable_attacks
    # Initialize attackers
    attackers = {}
    for attack in attacks:
        if attack != AllAttacks.REFERENCE_BASED:
            attackers[attack] = get_attacker(attack)(config, target_model)
        else:
            attackers[attack] = get_attacker(attack)(config, target_model, ref_models[config.ref_config.models[0]])

    return attackers

from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field
@dataclass
class AttackConfig(Serializable):
    model_name: str
    dataset:str
    model_mode: str