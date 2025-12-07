from MembershipFeature import PythiaFeature
import torch
from tqdm import tqdm
import os
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append('..')
from model_snapshot import PtPythiaSnapshots, model_dict
from pile_dataset import Pile
from MADataset import MADataset
from transformers.utils import logging
from util import *

logging.set_verbosity_error()
attack = sys.argv[1]
requested_mode = sys.argv[2].lower() if len(sys.argv) > 2 else "both"
model_name = model_dict[sys.argv[3]]
dataname = "pile"
pt_access = "open"
ft_access = "open"
feature_dim = 32

def extract_features_for_mode(training_mode: str):
    dataset = MADataset(dataname=dataname, mode=training_mode)
    dataloader = DataLoader(dataset, batch_size=10)
    snapshots = PtPythiaSnapshots(training_mode, sys.argv[3])
    extractor = PythiaFeature(snapshots.model_first, snapshots.model_second, snapshots.tokenizer, pt_access, ft_access)
    pred = []
    with torch.no_grad():
        for _, samples in enumerate(tqdm(dataloader, desc=f"{training_mode}")):
            samples = extractor.tokenizer.batch_decode(samples)
            scores = getattr(extractor, attack)(samples, feature_dim=feature_dim)
            pred.extend([p.cpu() for p in scores])
    torch.save(pred, f'output/pt_pt/{model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_{training_mode}_{feature_dim}.pt')
    del snapshots, extractor, pred
    torch.cuda.empty_cache()


if not os.path.isdir(f"output/pt_pt/{model_name}/{dataname}"):
    os.makedirs(f"output/pt_pt/{model_name}/{dataname}")

modes = ["target", "shadow"] if requested_mode == "both" else [requested_mode]
for mode in modes:
    if mode == 'shadow' and sys.argv[3] != 'pythia-1.4b':
        break
    extract_features_for_mode(mode)
