from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
class InstrumentDataset(Dataset):
    def __init__(self, folder, n=1000):
        self.folder = folder
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = np.load(os.path.join(self.folder, f"instrument_{i+1}_labels_tg.npz"))
        vol = np.zeros(d['shape'], np.uint8)
        c = d['coords']
        vol[c[:, 0], c[:, 1], c[:, 2]] = d['labels']
        return torch.from_numpy(vol)
