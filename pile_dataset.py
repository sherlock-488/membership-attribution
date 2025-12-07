import numpy as np
from torch.utils.data import Dataset
import dotenv
dotenv.load_dotenv()
PILE_PATH = dotenv.get_key(dotenv.find_dotenv(), 'PILE_PATH')
class Pile(Dataset):
    def __init__(self, mode='target'):

        index = 300 if mode == 'target' else 600
        dataset_1 = np.load(f'{PILE_PATH}/{(index-1)*100}-{index*100}-indicies-n1000-samples.npy')
        dataset_2 = np.load(f'{PILE_PATH}/{(index+99)*100}-{(index+100)*100}-indicies-n1000-samples.npy')
        dataset_3 = np.load(f'{PILE_PATH}/{(index+199)*100}-{(index+200)*100}-indicies-n1000-samples.npy')
        self.dataset = np.concatenate([dataset_1, dataset_2, dataset_3])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
class PilePerStep(Dataset):
    def __init__(self, step=1000):
        member = np.load(f'{PILE_PATH}/{step-100}-{step}-indicies-n1000-samples.npy')
        nonmember = np.load(f'{PILE_PATH}/test_tk.npy')
        self.dataset = np.concatenate([member, nonmember])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

if __name__ == '__main__':
    data = Pile(mode='target')
    print(len(data))
    import ipdb;ipdb.set_trace()
    