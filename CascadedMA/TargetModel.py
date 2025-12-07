from mimir.models import Model, ReferenceModel, LanguageModel
from typing import List
from mimir.config import ExperimentConfig
import numpy as np
from util import AttackConfig, SAVE_PATH
from collections import defaultdict
from mimir.custom_datasets import SEPARATOR
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch.nn.functional as F
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_snapshot import model_dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PythiaModel(LanguageModel):
    def __init__(self, config: ExperimentConfig, attack_config: AttackConfig,
                  revision: int = 1000, **kwargs):
        Model.__init__(self, config, **kwargs)
        self.config = config
        self.device = 'cuda'
        self.name = model_dict[attack_config.model_name]
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading model {self.name} at revision step{revision}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            return_dict=True,
            revision=f"step{revision}",
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.load_model_properties()

class LMWraper(LanguageModel):
    """
        Generic LM- used most often for target model
    """
    def __init__(self, config: ExperimentConfig, model, tokenizer, **kwargs):
        Model.__init__(self, config, **kwargs)
        self.device = self.config.env_config.device
        self.device_map = self.config.env_config.device_map
        self.name = config.base_model
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = 1024
        self.stride = 512