# sampling data according to a given index
import math
import numpy as np
from torch.utils.data import Dataset

class StaticLoader(Dataset):
    def __init__(self, dataset, selected_samples_index):
        self.dataset = dataset
        self.selected_samples_index = selected_samples_index        
        self.transform = dataset.transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index    

    def pruning_sampler(self):
        return StaticSampler(self)

    def no_prune(self):
        #samples = list(range(len(self.dataset)))
        samples = self.selected_samples_index
        np.random.shuffle(samples)
        return samples



class StaticSampler():
    def __init__(self, new_dataset):
        self.new_dataset = new_dataset
        self.seq = None
        self.reset()

    def reset(self):
        self.seq = self.new_dataset.no_prune()        
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self