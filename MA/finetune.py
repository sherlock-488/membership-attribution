import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
import transformers
import sys
from util import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from model_snapshot import model_dict
from MADataset import MADataset_FT_FT

# Optional pt_ft datasets
try:
    from pt_ft_util import load_train_data as ptft_load_train_data, SAVE_PATH as PTFT_SAVE_PATH
except ImportError:
    ptft_load_train_data = None
    PTFT_SAVE_PATH = SAVE_WEIGHTS_PATH

ptft_datasets = {"oloma", "mimir", "wikimia"}

model_name = model_dict[sys.argv[1]]
dataname = sys.argv[2]
training_mode = sys.argv[3]

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', 
                                             trust_remote_code=True,
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
config = LoraConfig(
    r=16,
    lora_alpha=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

save_root = PTFT_SAVE_PATH if dataname in ptft_datasets else SAVE_WEIGHTS_PATH
save_path_first = save_root + f'weights/ft_ft/{model_name}/{dataname}_{training_mode}_first'
save_path_second = save_root + f'weights/ft_ft/{model_name}/{dataname}_{training_mode}_second'

def run_ptft_branch():
    def guess_lora_target_modules(model):
        priors = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
            "query_key_value",
        ]
        leaf_names = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                leaf_names.add(name.split(".")[-1])
        targets = [p for p in priors if p in leaf_names]
        if not targets:
            fallback = set()
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(
                    kw in name for kw in ["attn","attention","mlp","feed_forward"]
                ):
                    fallback.add(name.split(".")[-1])
            targets = sorted(fallback)
        if not targets:
            raise RuntimeError("LoRA target modules auto-detection failed; please specify manually")
        return sorted(set(targets))

    def print_trainable_parameters(model):
        trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for _, p in model.named_parameters())
        pct = 100 * trainable / total if total else 0
        print(f"[LoRA] trainable params: {trainable} / {total} ({pct:.4f}%)")

    if ptft_load_train_data is None:
        raise ImportError("pt_ft_util not available for pt_ft dataset finetune")

    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
    base_model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    lora_targets = guess_lora_target_modules(base_model)
    print(f"[LoRA] target_modules = {lora_targets}")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_targets
    )
    base_model = get_peft_model(base_model, lora_cfg)
    print_trainable_parameters(base_model)

    save_path_second_local = PTFT_SAVE_PATH + f'weights/{model_name}/{dataname}_{training_mode}_second'
    data = ptft_load_train_data(dataname, training_mode, training_stage='second')
    data = data.map(lambda samples: tok(samples['text'], max_length=128, truncation=True), batched=True)

    trainer = transformers.Trainer(
        model=base_model,
        train_dataset=data['input_ids'],
        args=transformers.TrainingArguments(
            do_train=True,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            logging_strategy='epoch',
            num_train_epochs=2,
            save_strategy='no',
            learning_rate=1e-3,
            output_dir=f'weights/{model_name}/{dataname}_{training_mode}/'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tok, mlm=False)
    )
    trainer.train()
    base_model.save_pretrained(save_path_second_local)


def run_classic_branch():
    model_local = get_peft_model(model, config)

    def get_stage_data(stage):
        dataset = MADataset_FT_FT(dataname=dataname, mode=training_mode)
        return dataset.get_training_data(training_stage=stage)

    data = get_stage_data(stage='first')
    data = data.map(lambda samples: tokenizer(samples['text'], max_length=128, truncation=True), batched=True)
    trainer = transformers.Trainer(
        model=model_local, 
        train_dataset=data['input_ids'],
        args=transformers.TrainingArguments(
            do_train = True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            logging_strategy = 'epoch',
            num_train_epochs=5,
            save_strategy='no',
            learning_rate=1e-3,
            output_dir=save_root + f'weights/ft_ft/{model_name}'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    model_local.save_pretrained(save_path_first)

    config_local = PeftConfig.from_pretrained(save_path_first)
    model_local = AutoModelForCausalLM.from_pretrained(config_local.base_model_name_or_path, return_dict=True, device_map='auto', trust_remote_code=True, quantization_config=quantization_config)
    tokenizer_local = AutoTokenizer.from_pretrained(config_local.base_model_name_or_path, trust_remote_code=True)
    tokenizer_local.pad_token = tokenizer_local.eos_token
    model_local = PeftModel.from_pretrained(model_local, model_id=save_path_first, is_trainable=True)

    data = get_stage_data(stage='second')
    data = data.map(lambda samples: tokenizer_local(samples['text'], max_length=128, truncation=True), batched=True)
    trainer = transformers.Trainer(
        model=model_local, 
        train_dataset=data['input_ids'],
        args=transformers.TrainingArguments(
            do_train = True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            logging_strategy = 'epoch',
            num_train_epochs=5,
            save_strategy='no',
            learning_rate=1e-3,
            output_dir=save_root + f'weights/ft_ft/{model_name}'
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer_local, mlm=False)
    )
    trainer.train()
    model_local.save_pretrained(save_path_second)


if dataname in ptft_datasets:
    run_ptft_branch()
else:
    run_classic_branch()
