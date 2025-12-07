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
from TargetModel import TargetModel, PythiaModel
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pile_dataset import Pile, PilePerStep


parser = ArgumentParser(add_help=False)
parser.add_argument("--config", help="Path to attack config file", type=Path, default="config/mi.json")
parser.add_argument("--attack_config", type=Path, default="config/attack_config.json")
parser.add_argument("--step", type=int, default=1000)

args, remaining_argv = parser.parse_known_args()
step = args.step

# Attempt to extract as much information from config file as you can
config = ExperimentConfig.load(args.config)
attack_config = AttackConfig.load(args.attack_config)

# Also give user the option to provide config values over CLI
parser = ArgumentParser(parents=[parser])
parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
args = parser.parse_args(remaining_argv)
config: ExperimentConfig = args.exp_config

env_config: EnvironmentConfig = config.env_config
neigh_config: NeighborhoodConfig = config.neighborhood_config
ref_config: ReferenceConfig = config.ref_config
openai_config: OpenAIConfig = config.openai_config

base_model = PythiaModel(config, attack_config, revision=step)
ref_models = {model: ReferenceModel(config, model) for model in ref_config.models}


attackers_dict = get_attackers(base_model, ref_models, config)
if attack_config.dataset == 'pile': 
    dataset = PilePerStep(step=step)
    env_config.pretokenized = True

from sklearn.metrics import roc_auc_score
import json
dataloader = DataLoader(dataset, batch_size=1)
pred = []
auc_results = {}
output_dir = f"output/{config.base_model}/{attack_config.dataset}"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

results_path = f"{output_dir}/auc_step_{step}.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        auc_results = json.load(f)
    print(f"Loaded existing AUC results from {results_path}")
for k in attackers_dict.keys():
    attack_name = str(k)
    if attack_name in auc_results:
        print(f"Skipping attack {attack_name} (already in results)")
        continue
    pred = []
    for idx, samples in enumerate(tqdm(dataloader)):
        if env_config.pretokenized:
            texts = attackers_dict[k].target_model.tokenizer.batch_decode(samples)
            score = attackers_dict[k].attack(texts, probs=None, tokens=samples.numpy())
        pred.append(score)
    scores = torch.cat([torch.as_tensor(s).flatten() for s in pred]).cpu().numpy()
    labels = np.concatenate([np.zeros(1000), np.ones(1000)])
    scores = np.nan_to_num(scores[: len(labels)], nan=0.0, posinf=0.0, neginf=0.0)
    auc = roc_auc_score(labels, scores[: len(labels)])
    auc_results[attack_name] = float(auc)
    print(f"Step: {step}, Attack: {attack_name}, AUC: {auc:.4f}")

with open(results_path, "w") as f:
    json.dump(auc_results, f, indent=2)
print(f"AUC results saved to {results_path}")
