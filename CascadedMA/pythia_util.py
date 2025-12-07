import torch
import sys
sys.path.append('..')
from util import *
import torch
from tqdm import tqdm
import os
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

def _log_value(probs, small_value=1e-30):
    probs = probs.float()
    return -torch.log(torch.clamp(probs, min=small_value))

def prob_diff(prob_1, prob_2):

    delta_logits = _log_value(prob_1)-_log_value(prob_2)
    contrastive_prob = softmax(delta_logits, -1)

    return contrastive_prob

def prob_diff_weight(prob_1, prob_2, label):
    prob_1 = prob_1.view(label.shape[0], -1)
    prob_2 = prob_2.view(label.shape[0], -1)
    
    prob_1_true = prob_1[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)
    prob_2_true = prob_2[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)

    token_masks = _log_value(prob_2_true)-_log_value(prob_1_true)

    return token_masks

def get_last_prob(model, tokens):
    outputs = model(**tokens)
    last_layer_prob = softmax(outputs.logits[..., :-1, :].float(), -1)
    return last_layer_prob

def entr_sum(probs):
    return -torch.sum(torch.mul(probs, _log_value(probs)), -1)

def entr_w(samples, tokenizer, model1, model2):
    ret_entr = []
    with torch.no_grad():
        tokens = tokenizer(samples, return_tensors='pt', padding='max_length', max_length=128, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        last_prob = get_last_prob(model2, tokens)
        last_prob_ori = get_last_prob(model1, tokens)
        probs = last_prob
        
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
    
        token_masks = prob_diff_weight(last_prob_ori, last_prob, shift_labels)
    
        mask = (token_masks.view(shift_masks.shape))*shift_masks

        entr = entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        entr = -torch.sum(torch.mul(mask,entr), -1)/torch.sum(shift_masks, -1)
        ret_entr.extend([e.cpu() for e in entr])
    return ret_entr
