# finetune.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import transformers
import sys
from util import *

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
        raise RuntimeError("LoRA 目标层自动识别失败，请手动指定。")
    return sorted(set(targets))

def print_trainable_parameters(model):
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    total = sum(p.numel() for _, p in model.named_parameters())
    pct = 100 * trainable / total if total else 0
    print(f"[LoRA] trainable params: {trainable} / {total} ({pct:.4f}%)")

model_name = model_dict[sys.argv[1]]
dataname = sys.argv[2]       
training_mode = sys.argv[3]   

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)

model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_targets = guess_lora_target_modules(model)
print(f"[LoRA] target_modules = {lora_targets}")

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules=lora_targets
)
model = get_peft_model(model, lora_cfg)
print_trainable_parameters(model)

save_path_second = SAVE_PATH + f'weights/{model_name}/{dataname}_{training_mode}_second'

# === Second 数据：Y ∪ Z ===
data = load_train_data(dataname, training_mode, training_stage='second')
data = data.map(lambda samples: tokenizer(samples['text'], max_length=128, truncation=True), batched=True)
# all
# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=data['input_ids'],
#     args=transformers.TrainingArguments(
#         do_train=True,
#         per_device_train_batch_size=8,
#         gradient_accumulation_steps=4,
#         logging_strategy='epoch',
#         num_train_epochs=2,            
#         save_strategy='no',
#         learning_rate=1e-3,
#         # lr_scheduler_type="constant",
#         # warmup_steps=0,         
#         tf32=True,
#         output_dir=f'weights/{model_name}/{dataname}_{training_mode}/'
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
# )
# llama2
trainer = transformers.Trainer(
    model=model,
    train_dataset=data['input_ids'],
    args=transformers.TrainingArguments(
        do_train=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        logging_strategy='epoch',
        num_train_epochs=2,            
        save_strategy='no',
        learning_rate=1e-3,
        # lr_scheduler_type="constant",
        # warmup_steps=0,         
        tf32=True,
        output_dir=f'weights/{model_name}/{dataname}_{training_mode}/'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
model.save_pretrained(save_path_second)
