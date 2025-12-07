"""
    ReCaLL Attack: https://github.com/ruoyuxie/recall/
"""
import torch 
import numpy as np
from mimir.attacks.recall import ReCaLLAttack
from mimir.models import Model
from mimir.config import ExperimentConfig

class ConReCaLLAttack(ReCaLLAttack):

    #** Note: this is a suboptimal implementation of the ReCaLL attack due to necessary changes made to integrate it alongside the other attacks
    #** for a better performing version, please refer to: https://github.com/ruoyuxie/recall 
    
    def __init__(self, config: ExperimentConfig, target_model: Model):
        super().__init__(config, target_model)
        self.prefix = None

    @torch.no_grad()
    def _attack(self, document, probs, tokens = None, **kwargs):
        # recall_dict = {"prefix":nonmember_prefix, "num_shots":1, "avg_length":300}        
        recall_dict: dict = kwargs.get("recall_dict", None)

        nonmember_prefix = recall_dict.get("prefix")
        num_shots = recall_dict.get("num_shots")
        avg_length = recall_dict.get("avg_length")
        member_prefix = recall_dict.get("member_prefix")

        assert nonmember_prefix, "nonmember_prefix should not be None or empty"
        assert num_shots, "num_shots should not be None or empty"
        assert avg_length, "avg_length should not be None or empty"

        lls = self.target_model.get_ll(document, probs = probs, tokens = tokens)
        ll_nonmember = self.get_conditional_ll(nonmember_prefix = nonmember_prefix, text = document,
                                                num_shots = num_shots, avg_length = avg_length,
                                                  tokens = tokens)
        
        ll_member = self.get_conditional_ll(nonmember_prefix = member_prefix, text = document,
                                                num_shots = num_shots, avg_length = avg_length,
                                                  tokens = tokens)
        conrecall = (ll_nonmember - ll_member)/lls
        # recall = ll_nonmember / lls


        assert not np.isnan(conrecall)
        return conrecall
    