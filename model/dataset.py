import numpy as np
from torch.utils.data import Dataset


class ConfigurationDataset(Dataset):
    '''
    the dataset for the Configuration Network
    '''
    def __init__(self, input_path, label_path):
        self.input = np.asarray(np.load(input_path), dtype=np.float32)
        self.label = np.asarray(np.load(label_path), dtype=np.int32)
    
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]


class ReachDataset(Dataset):
    '''
    the dataset for the Reachability Network
    '''
    def __init__(self, path):
        data = np.asarray(np.load(path), dtype=np.float32)
        if(data.shape[1] == 9):
            self.data = np.zeros((data.shape[0], 10))
            self.data[:,:-1] = data.copy()
            self.data[:,-1] = 1
        else:
            self.data = data
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:-1], self.data[idx,-1]