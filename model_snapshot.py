from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model_dict = {
    'gpt-neo': 'EleutherAI/gpt-neo-125m',
    'gpt-j': 'EleutherAI/gpt-j-6B',
    'opt': 'facebook/opt-350m',
    'pythia': 'EleutherAI/pythia-1.4b-deduped',
    'pythia-410m': 'EleutherAI/pythia-410m-deduped',
    'pythia-1b': 'EleutherAI/pythia-1b-deduped',
    'pythia-1.4b': 'EleutherAI/pythia-1.4b-deduped',
    'pythia-6.9b': 'EleutherAI/pythia-6.9b-deduped',
    'pythia-12b': 'EleutherAI/pythia-12b-deduped',
    'llama3': 'meta-llama/Llama-3.2-1B',
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'qwen3': "Qwen/Qwen3-4B",
    'olmo': 'allenai/OLMo-1B-hf',
}
SAVE_WEIGHTS_PATH = "../"
class PtPythiaSnapshots:
    def __init__(self, mode: str, model_name: str):
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.name = model_dict[model_name]
        self.load_pretraining_pythia(mode)
    
    def load_pretraining_pythia(self, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if mode == 'target':
            print(f"Loading model {self.name} for {mode} mode")
            self.model_first = AutoModelForCausalLM.from_pretrained(self.name, return_dict=True, revision="step30000", device_map='auto', quantization_config=self.quantization_config, weights_only=False)
            self.model_second = AutoModelForCausalLM.from_pretrained(self.name, return_dict=True, revision="step40000", device_map='auto', quantization_config=self.quantization_config,weights_only=False)
        elif mode == 'shadow':
            print(f"Loading model pythia-1.4b for {mode} mode")
            self.model_first = AutoModelForCausalLM.from_pretrained(model_dict['pythia-1.4b'], return_dict=True, revision="step60000", device_map='auto', quantization_config=self.quantization_config, weights_only=False)
            self.model_second = AutoModelForCausalLM.from_pretrained(model_dict['pythia-1.4b'], return_dict=True, revision="step70000", device_map='auto', quantization_config=self.quantization_config,weights_only=False)

class FtFTSnapshots:
    def __init__(self, mode: str, model_name: str, dataname: str):
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.name = model_dict[model_name]
        self.dataname = dataname
        self.load_finetuned_model(mode)
    
    def load_finetuned_model(self, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        save_path_first = SAVE_WEIGHTS_PATH + f'weights/ft_ft/{self.name}/{self.dataname}_{mode}_first'
        save_path_second = SAVE_WEIGHTS_PATH + f'weights/ft_ft/{self.name}/{self.dataname}_{mode}_second'
        print(f"Loading finetuned model from {save_path_first} and {save_path_second}")
        self.model_first = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(self.name, return_dict=True, device_map='auto', quantization_config=self.quantization_config),
            model_id=save_path_first,
            is_trainable=False
        )
        self.model_second = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(self.name, return_dict=True, device_map='auto', quantization_config=self.quantization_config),
            model_id=save_path_second,
            is_trainable=False
        )
