# python train_attacker.py ma_w pythia-410m pythia-1.4b pt_pt pile
# python train_attacker.py ma_w pythia-1b pythia-1.4b pt_pt pile
# python train_attacker.py ma_w pythia-1.4b pythia-1.4b pt_pt pile
# python train_attacker.py ma_w pythia-6.9b pythia-1.4b pt_pt pile
# python train_attacker.py ma_w pythia-12b pythia-1.4b pt_pt pile

# python evaluate_attacker.py ma_w pythia-12b pythia-1.4b pt_pt pile


# python finetune.py qwen3 fineweb target
# python finetune.py qwen3 fineweb shadow

python finetune.py llama3 fineweb target
python finetune.py llama3 fineweb shadow