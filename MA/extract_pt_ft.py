# extract_feature.py
from pt_ft_util import *
from MembershipFeature import MembershipFeature, OLoMAFeature
import torch
from tqdm import tqdm
import os
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from transformers.utils import logging

logging.set_verbosity_error()

attack = sys.argv[1]
model_name = model_dict[sys.argv[2]]
dataname = sys.argv[3]       # 'oloma' / 'mimir' / 'wikimia' / ...
training_mode = sys.argv[4]  # 'target' or 'shadow'
pt_access = sys.argv[5]
ft_access = sys.argv[6]
feature_dim = int(sys.argv[7])

token_feature_mode = sys.argv[8] if len(sys.argv) > 8 else "topk"

dataset = load_eval_data(dataname, training_mode)
dataloader = DataLoader(dataset, batch_size=10)

# 统一输出目录：用 SAVE_PATH
out_dir = os.path.join(
    SAVE_PATH,
    f"output/{model_name}/{dataname}"
)
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"{attack}_{pt_access}_{ft_access}_{training_mode}_{feature_dim}.pt"
)

pred = []

with torch.no_grad():
    # ===== OLOMA / MIMIR / WIKIMIA：M1 = base 模型，M2 = base + LoRA(second) =====
    if dataname in ['oloma', 'mimir', 'wikimia']:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # M1: 纯 base 模型（公开预训练）
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            trust_remote_code=True,
            quantization_config=quantization_config
        )

        # M2: 另加载一份 base，再挂 LoRA(second)
        base_model_for_second = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            trust_remote_code=True,
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model_id_second = SAVE_PATH + f'weights/{model_name}/{dataname}_{training_mode}_second'
        second_model = PeftModel.from_pretrained(
            base_model_for_second,
            model_id=model_id_second,
            adapter_name="second"
        )

        # 防止误修改 base 权重
        second_model.base_model.requires_grad_(False)

        extractor = OLoMAFeature(base_model, second_model, tokenizer, pt_access, ft_access, max_length=128)

        for _, samples in enumerate(tqdm(dataloader)):
            scores = getattr(extractor, attack)(samples, feature_dim=feature_dim)
            pred.extend([p.cpu() for p in scores])
            torch.save(pred, out_path)

    # ===== 其他数据集（agnews / onion 等），保留原来的 MembershipFeature 路线 =====
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_id_first = SAVE_PATH + f'weights/{model_name}/{dataname}_{training_mode}_first'
        model_id_second = SAVE_PATH + f'weights/{model_name}/{dataname}_{training_mode}_second'

        config = PeftConfig.from_pretrained(model_id_first)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            device_map='auto',
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(model, model_id=model_id_first, adapter_name="first")
        model.load_adapter(model_id_second, adapter_name="second")

        extractor = MembershipFeature(model, tokenizer, pt_access, ft_access, max_length=128)

        for _, samples in enumerate(tqdm(dataloader)):
            scores = getattr(extractor, attack)(samples, feature_dim=feature_dim)
            pred.extend([p.cpu() for p in scores])
            torch.save(pred, out_path)
