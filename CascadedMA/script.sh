# for step in $(seq 5000 5000 95000); do
#     python attack_pile_step.py --step "$step"
# done

# for step in $(seq 1000 2000 95000); do
#     echo "Running incremental step: $step"
#     python MA_pile_step.py --incremental_step "$step" --step 5000
# done

# python evaluate_pt_pt.py zlib pile pythia-1b pythia-1.4b
# python evaluate_pt_pt.py zlib pile pythia-1.4b pythia-1.4b
# python evaluate_pt_pt.py zlib pile pythia-6.9b pythia-1.4b
# python evaluate_pt_pt.py zlib pile pythia-12b pythia-1.4b

python attack_ft_ft.py --attack_config config/attack_config_ft_ft.json
python attack_ft_ft.py --attack_config config/attack_config_ft_ft_qwen.json


# python evaluate_ft_ft.py zlib agnews llama3 llama3
# python evaluate_ft_ft.py zlib agnews llama3 llama3

