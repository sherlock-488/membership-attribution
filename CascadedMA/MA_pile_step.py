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
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pile_dataset import Pile, PilePerStep


parser = ArgumentParser(add_help=False)
parser.add_argument("--config", help="Path to attack config file", type=Path, default="config/mi.json")
parser.add_argument("--attack_config", type=Path, default="config/attack_config.json")
parser.add_argument("--step", type=int, default=1000)
parser.add_argument("--incremental_step", type=int, default=1000)

args, remaining_argv = parser.parse_known_args()
step = args.step
incremental_step = args.incremental_step
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

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
tokenizer.pad_token = tokenizer.eos_token

model_0 = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{attack_config.model_name}", return_dict=True, revision=f"step{step}",
                                            device_map='auto', quantization_config=quantization_config)
model_1 = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{attack_config.model_name}", return_dict=True, revision=f"step{step+incremental_step}",
                                            device_map='auto', quantization_config=quantization_config)

if attack_config.dataset == 'pile': 
    dataset = PilePerStep(step=step)
    env_config.pretokenized = True
dataloader = DataLoader(dataset, batch_size=20)
pred = []
output_dir = f"output/{attack_config.model_name}/{attack_config.dataset}"
results_path = os.path.join(output_dir, "ma_auc.json")
if not os.path.isdir(output_dir): 
    os.makedirs(output_dir) 
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        auc_results = json.load(f)
    print(f"Loaded existing AUC results from {results_path}")
else:
    auc_results = {}


from pythia_util import entr_w
from sklearn.metrics import roc_auc_score
import numpy as np

with torch.no_grad():
    pred = []
    for idx, samples in enumerate(tqdm(dataloader)):
        if env_config.pretokenized:
            texts = tokenizer.batch_decode(samples)
            score = entr_w(texts, tokenizer, model_0, model_1)
            # score = attackers_dict[k].attack(None, probs=attackers_dict[k].target_model.get_probabilities(None, tokens=samples.numpy()))
        pred.append(score)
    scores = torch.cat([torch.as_tensor(s).flatten() for s in pred]).cpu().numpy()
    labels = np.concatenate([np.ones(1000), np.zeros(1000)])
    auc = roc_auc_score(labels, scores[: len(labels)])
    print(f"AUC: {auc:.4f}")
    step_key = str(step)
    inc_key = str(incremental_step)
    if step_key not in auc_results:
        auc_results[step_key] = {}
    auc_results[step_key][inc_key] = float(auc)
    with open(results_path, "w") as f:
        json.dump(auc_results, f, indent=2)
    print(f"AUC results saved to {results_path}")
