from simple_parsing import ArgumentParser
from pathlib import Path
from util import *
import os
from tqdm import tqdm
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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MADataset import MADataset, MADataset_FT_FT
from model_snapshot import PtPythiaSnapshots, model_dict, FtFTSnapshots
from TargetModel import LMWraper
from ConReCaLL import ConReCaLLAttack

parser = ArgumentParser(add_help=False)
parser.add_argument("--config", help="Path to attack config file", type=Path, default="config/mi.json")
parser.add_argument("--attack_config", type=Path, default="config/attack_config_ft_ft.json")

args, remaining_argv = parser.parse_known_args()
config = ExperimentConfig.load(args.config)
attack_config = AttackConfig.load(args.attack_config)
modelname = model_dict[attack_config.model_name]
parser = ArgumentParser(parents=[parser])
parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
args = parser.parse_args(remaining_argv)
config: ExperimentConfig = args.exp_config

env_config: EnvironmentConfig = config.env_config
ref_config: ReferenceConfig = config.ref_config

def attack(base_model, ref_models, dataset, rank):
    attackers_dict = get_attackers(base_model, ref_models, config)
    if 'conrecall' in config.blackbox_attacks:
        attackers_dict['conrecall'] = ConReCaLLAttack(config, base_model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for k in attackers_dict.keys():
        pred = []
        if rank.startswith('second') and k != 'loss':
            break
        for idx, samples in enumerate(tqdm(dataloader, desc=f"Extracting features for {k} attack")):
            if k == 'recall' or k == 'conrecall':
                nonmember_prefix = dataset[-1:]['text']
                member_prefix = dataset[2500:2501]['text']
                recall_dict = {"prefix":nonmember_prefix, "num_shots":1, 
                               "avg_length":300,
                               "member_prefix":member_prefix}
                document = samples['text']
                score = attackers_dict[k].attack(document, probs=None, recall_dict=recall_dict)
            else:
                document = samples['text']
                score = attackers_dict[k].attack(document, tokens=None, probs=None)
            pred.append(score)
        torch.save(pred, f'output/ft_ft/{modelname}/{attack_config.dataset}/{k}_{rank}.pt')

def extract_features_for_mode(training_mode: str):
    dataset = MADataset_FT_FT(dataname=attack_config.dataset, mode=training_mode)
    snapshots = FtFTSnapshots(training_mode, attack_config.model_name, attack_config.dataset)

    base_model = LMWraper(config, snapshots.model_first, snapshots.tokenizer)
    ref_models = {model: ReferenceModel(config, model) for model in ref_config.models}
    attack(base_model, ref_models, dataset, f'first_{training_mode}')
    
    del base_model, ref_models
    torch.cuda.empty_cache()

    base_model = LMWraper(config, snapshots.model_second, snapshots.tokenizer)
    ref_models = {model: ReferenceModel(config, model) for model in ref_config.models}
    attack(base_model, ref_models, dataset, f'second_{training_mode}')

    del snapshots, base_model, ref_models
    torch.cuda.empty_cache()

if not os.path.isdir(f"output/ft_ft/{modelname}/{attack_config.dataset}"): 
    os.makedirs(f"output/ft_ft/{modelname}/{attack_config.dataset}") 


modes = ["target", "shadow"]
for mode in modes:
    extract_features_for_mode(mode)