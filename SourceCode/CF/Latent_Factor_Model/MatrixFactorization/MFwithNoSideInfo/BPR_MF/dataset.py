from torch.utils.data import Dataset
import numpy as np

class BprMFData(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_user = len(data)
    def __len__(self):
        return self.num_user
    def __getitem__(self,idx):
        u = list(self.data.keys())[idx]
        i = self.data[u][0][np.random.randint(0,len(self.data[u][0]))]
        j = self.data[u][1][np.random.randint(0,len(self.data[u][1]))]
        return (u,i,j)