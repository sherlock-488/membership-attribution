import torch
from torch.nn.functional import softmax
from tqdm import tqdm

class MembershipFeature():
    def __init__(self, model, tokenizer, pt='open', ft='open', max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.first_query = self.open_probe if pt == 'open' else self.close_probe
        self.second_query = self.open_probe if ft == 'open' else self.close_probe
        self.max_length = max_length

    def _log_value(self, probs, small_value=1e-30):
        probs = probs.float()
        return -torch.log(torch.clamp(probs, min=small_value))
    
    def open_probe(self, model, tokens):
        outputs = model(**tokens)
        last_layer_prob = softmax(outputs.logits[..., :-1, :].float(), -1)
        return last_layer_prob

    def close_probe(self, model, tokens, query_time=100):
        preds = []
        for idx in range(1, tokens.input_ids.shape[-1]):
            candidates = []
            gt = model.generate(tokens.input_ids[:, :idx], max_new_tokens=1, top_p=0.9, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, return_dict_in_generate=True, output_scores=True)
            probs = torch.nn.functional.softmax(gt.scores[0], dim=-1)
            for _ in range(query_time):
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                candidates.append(next_token)
            candidates = torch.stack(candidates, 0)
            y_onehot = torch.nn.functional.one_hot(candidates, num_classes=probs.shape[-1])
            y_pred = torch.sum(y_onehot.float(), 0)/query_time
            preds.append(y_pred)
        return torch.stack(preds, 1)
    
    def prob_diff(self, prob_1, prob_2):

        delta_logits = self._log_value(prob_2)-self._log_value(prob_1)
        contrastive_prob = softmax(delta_logits, -1)

        return contrastive_prob

    def prob_diff_weight(self, prob_1, prob_2, label):
        prob_1 = prob_1.view(label.shape[0], -1)
        prob_2 = prob_2.view(label.shape[0], -1)
        
        prob_1_true = prob_1[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)
        prob_2_true = prob_2[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)

        token_masks = self._log_value(prob_2_true)-self._log_value(prob_1_true)

        return token_masks
    
    def m_entr_sum(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(1-probs)
        modified_probs = torch.clone(probs)
        modified_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)] = reverse_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)]
        modified_log_probs = torch.clone(log_reverse_probs)
        modified_log_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)] = log_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)]
        return -torch.sum(torch.mul(modified_probs, modified_log_probs), axis=1)

    def entr_sum(self, probs):
        return -torch.sum(torch.mul(probs, self._log_value(probs)), -1)

    def ma_diff_w(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples['text'], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        self.model.set_adapter("second")
        last_prob = self.second_query(self.model, tokens)
        self.model.set_adapter("first")
        last_prob_ori = self.first_query(self.model, tokens)
        probs = self.prob_diff(last_prob_ori, last_prob)
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
        
        token_masks = self.prob_diff_weight(last_prob_ori, last_prob, shift_labels)
        mask = (token_masks.view(shift_masks.shape))*shift_masks
        token_preds = torch.mul(torch.sort(probs, -1, descending=True).values, mask.unsqueeze(-1))[:, :, :feature_dim-2]

        entr = self.entr_sum(last_prob.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(last_prob.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)

        entr = torch.mul(mask,entr).unsqueeze(-1)
        m_entr = torch.mul(mask,m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds], -1)

        return sentence_preds

    def ma(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples['text'], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        self.model.set_adapter("second")
        last_prob = self.second_query(self.model, tokens)
        probs = last_prob
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
    
        mask = shift_masks
        token_preds = torch.mul(torch.sort(probs, -1, descending=True).values, mask.unsqueeze(-1))[:, :, :feature_dim-2]

        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(mask,entr).unsqueeze(-1)
        m_entr = torch.mul(mask,m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds], -1)

        return sentence_preds

    def ma_w(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples['text'], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        self.model.set_adapter("second")
        last_prob = self.second_query(self.model, tokens)
        self.model.set_adapter("first")
        last_prob_ori = self.first_query(self.model, tokens)
        probs = last_prob_ori
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
        
        token_masks = self.prob_diff_weight(last_prob_ori, last_prob, shift_labels)
    
        mask = (token_masks.view(shift_masks.shape))*shift_masks
        token_preds = torch.mul(torch.sort(probs, -1, descending=True).values, mask.unsqueeze(-1))[:, :, :feature_dim-2]

        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(mask,entr).unsqueeze(-1)
        m_entr = torch.mul(mask,m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds], -1)

        return sentence_preds
    
    def ma_w_new(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples['text'], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        self.model.set_adapter("second")
        last_prob = self.second_query(tokens)
        self.model.set_adapter("first")
        last_prob_ori = self.first_query(tokens)
        probs = last_prob_ori
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
        


        scores_per_token = torch.mul(-probs, self._log_value(probs)-self._log_value(last_prob))

        weights_per_token = self._log_value(probs)-self._log_value(last_prob)
        weights_per_token = weights_per_token.view(shift_labels.size(0),-1)
        true_weights_per_token = weights_per_token[torch.arange(shift_labels.size(0)).unsqueeze(1), shift_labels.unsqueeze(1)].view(shift_masks.shape[0], -1, 1)
        
        

        scores_per_token = scores_per_token.view(shift_labels.size(0),-1)
        true_scores_per_token = scores_per_token[torch.arange(shift_labels.size(0)).unsqueeze(1), shift_labels.unsqueeze(1)]
        scores_per_token[torch.arange(shift_labels.size(0)).unsqueeze(1), shift_labels.unsqueeze(1)] = torch.max(scores_per_token)
        scores_per_token = torch.sort(scores_per_token, -1, descending=True).values[:, :feature_dim-3]



        # token_preds = torch.cat([true_scores_per_token, torch.zeros_like(scores_per_token)], -1).view(shift_masks.shape[0], -1, feature_dim-2)
        token_preds = true_scores_per_token.view(shift_masks.shape[0], -1, 1)


        token_masks = self.prob_diff_weight(last_prob_ori, last_prob, shift_labels)

        mask = (token_masks.view(shift_masks.shape))*shift_masks
        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(mask,entr).unsqueeze(-1)
        m_entr = torch.mul(mask,m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds, true_weights_per_token, torch.zeros(shift_masks.shape[0], shift_masks.shape[1], feature_dim-4).to('cuda')], -1)

        return sentence_preds
    
    def ma_all(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples['text'], return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        self.model.set_adapter("second")
        last_prob = self.second_query(self.model, tokens)
        self.model.set_adapter("first")
        last_prob_ori = self.first_query(self.model, tokens)
        probs = last_prob_ori
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
        


        scores_per_token = torch.mul(-probs, self._log_value(probs)-self._log_value(last_prob))

        weights_per_token = self._log_value(probs)-self._log_value(last_prob)
        weights_per_token = weights_per_token.view(shift_labels.size(0),-1)
        true_weights_per_token = weights_per_token[torch.arange(shift_labels.size(0)).unsqueeze(1), shift_labels.unsqueeze(1)].view(shift_masks.shape[0], -1, 1)

        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(shift_masks,entr).unsqueeze(-1)
        m_entr = torch.mul(shift_masks,m_entr).unsqueeze(-1)
        probs = torch.mul(shift_masks.unsqueeze(-1),probs)
        probs = torch.sort(probs, -1, descending=True).values[:, :, :feature_dim-2]
        preds_1 = torch.cat([entr, m_entr, probs], -1)

        probs = last_prob
        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(shift_masks,entr).unsqueeze(-1)
        m_entr = torch.mul(shift_masks,m_entr).unsqueeze(-1)
        probs = torch.mul(shift_masks.unsqueeze(-1),probs)
        probs = torch.sort(probs, -1, descending=True).values[:, :, :feature_dim-2]
        preds_2 = torch.cat([entr, m_entr, probs], -1)

        return preds_1, preds_2, true_weights_per_token
    

class PythiaFeature(MembershipFeature):
    def __init__(self, model_0, model_1, tokenizer, pt='open', ft='open', max_length=300):
        super().__init__(None, tokenizer, pt, ft, max_length)
        self.model_0 = model_0
        self.model_1 = model_1
        self.second_query = self.open_probe if ft == 'open' else self.close_probe
    
    def ma_w(self, samples, feature_dim=32):
        tokens = self.tokenizer(samples, return_tensors='pt', padding='longest', max_length=self.max_length, truncation=True).to('cuda')
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        last_prob = self.second_query(self.model_1, tokens)
        last_prob_ori = self.first_query(self.model_0, tokens)
        probs = last_prob_ori
        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]
        
        token_masks = self.prob_diff_weight(last_prob_ori, last_prob, shift_labels)
    
        mask = (token_masks.view(shift_masks.shape))*shift_masks
        token_preds = torch.mul(torch.sort(probs, -1, descending=True).values, mask.unsqueeze(-1))[:, :, :feature_dim-2]

        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(mask,entr).unsqueeze(-1)
        m_entr = torch.mul(mask,m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds], -1)

        return sentence_preds


class OLoMAFeature(MembershipFeature):
    """
    Feature extractor used in the pt_ft pipeline (base + second model).
    """
    def __init__(self, base_model, second_model, tokenizer, pt='open', ft='open', max_length=128):
        super().__init__(None, tokenizer, pt, ft, max_length)
        self.base_model = base_model
        self.second_model = second_model

    def ma_w(self, samples, feature_dim=32):
        tokens = self.tokenizer(
            samples['text'],
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        ).to('cuda')

        shift_labels = tokens.input_ids[..., 1:].reshape(-1)
        shift_masks = tokens.attention_mask[..., 1:]

        last_prob = self.open_probe(self.second_model, tokens)
        last_prob_ori = self.open_probe(self.base_model, tokens)
        probs = last_prob_ori

        token_masks = self.prob_diff_weight(last_prob_ori, last_prob, shift_labels)
        mask = (token_masks.view(shift_masks.shape)) * shift_masks

        token_preds = torch.mul(
            torch.sort(probs, -1, descending=True).values,
            mask.unsqueeze(-1)
        )[:, :, :feature_dim-2]

        entr = self.entr_sum(probs.view(shift_labels.shape[0], -1)).view(shift_masks.shape)
        m_entr = self.m_entr_sum(probs.view(shift_labels.shape[0], -1), shift_labels).view(shift_masks.shape)
        entr = torch.mul(mask, entr).unsqueeze(-1)
        m_entr = torch.mul(mask, m_entr).unsqueeze(-1)
        sentence_preds = torch.cat([entr, m_entr, token_preds], -1)
        return sentence_preds
