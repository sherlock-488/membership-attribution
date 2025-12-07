from util import *
from MembershipFeature import MembershipFeature
import torch
from tqdm import tqdm
import os
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append('..')
from model_snapshot import model_dict
from transformers.utils import logging
from MADataset import MADataset_FT_FT

logging.set_verbosity_error()

def extract_features_for_mode(training_mode: str, attack: str, model_name: str, dataname: str, pt_access: str, ft_access: str, feature_dim: int):
    model_id_first = SAVE_WEIGHTS_PATH + f'weights/ft_ft/{model_name}/{dataname}_{training_mode}_first'
    model_id_second = SAVE_WEIGHTS_PATH + f'weights/ft_ft/{model_name}/{dataname}_{training_mode}_second'
    dataset = MADataset_FT_FT(dataname=dataname, mode=training_mode)
    dataloader = DataLoader(dataset, batch_size=10)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    config = PeftConfig.from_pretrained(model_id_first)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto', quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, model_id=model_id_first, adapter_name="first")
    model.load_adapter(model_id_second, adapter_name="second")

    extractor = MembershipFeature(model, tokenizer, pt_access, ft_access)
    if not os.path.isdir(f"output/ft_ft/{model_name}/{dataname}"): 
        os.makedirs(f"output/ft_ft/{model_name}/{dataname}") 
    pred = []
    with torch.no_grad():
        for idx, samples in enumerate(tqdm(dataloader, desc=f"{training_mode}")):
            scores = getattr(extractor, attack)(samples, feature_dim=feature_dim)
            pred.extend([p.cpu() for p in scores])
    torch.save(pred, f'output/ft_ft/{model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_{training_mode}_{feature_dim}.pt')
    del extractor, pred, model, tokenizer, dataloader, dataset
    torch.cuda.empty_cache()

if __name__ == '__main__':
    attack = sys.argv[1]
    requested_mode = sys.argv[2].lower() if len(sys.argv) > 2 else "both"
    model_name = model_dict[sys.argv[3]]
    pt_access = 'open'
    ft_access = 'open'
    feature_dim = 32
    dataname = sys.argv[4]

    if not os.path.isdir(f"output/ft_ft/{model_name}/{dataname}"):
        os.makedirs(f"output/ft_ft/{model_name}/{dataname}")

    modes = ["target", "shadow"] if requested_mode == "both" else [requested_mode]
    for mode in modes:
        extract_features_for_mode(mode, attack, model_name, dataname, pt_access, ft_access, feature_dim)
