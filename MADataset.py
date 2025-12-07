from pile_dataset import *
from datasets import load_dataset
from itertools import islice

class MADataset(Dataset):
    def __init__(self, dataname, mode='target'):
        self.dataname = dataname
        if dataname == 'pile':
            self.dataset = Pile(mode=mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class MADataset_FT_FT(MADataset):
    def __init__(self, dataname, mode='target'):
        super().__init__(dataname, mode)
        if dataname == 'fineweb':
            self._load_fineweb(mode=mode)
        if dataname == 'agnews':
            self._load_agnews(mode=mode)

    def _load_fineweb(self, mode):
        if mode == 'target':
            self.dataset = load_dataset("HuggingFaceFW/fineweb", 
                                        name="sample-10BT", 
                                        split="train[:10000]").select_columns('text')
        elif mode == 'shadow':
            self.dataset = load_dataset("HuggingFaceFW/fineweb", 
                                        name="sample-10BT", 
                                        split="train[10000:20000]").select_columns('text')
    
    def _load_agnews(self, mode):
        if mode == 'target':
            self.dataset = load_dataset("fancyzhx/ag_news", 
                                        split="train[:10000]")
        elif mode == 'shadow':
            self.dataset = load_dataset("fancyzhx/ag_news", 
                                        split="train[10000:20000]")
        self.dataset = self.dataset.shuffle(seed=2023).select_columns('text')

    def get_training_data(self, training_stage):
        if training_stage == 'first':
            return self.dataset.select(range(0, 5000))
        elif training_stage == 'second':
            return self.dataset.select(range(2500, 7500))
        
if __name__ == '__main__':
    data = MADataset_FT_FT(dataname='fineweb', mode='target')
    print(len(data))
    import ipdb;ipdb.set_trace()