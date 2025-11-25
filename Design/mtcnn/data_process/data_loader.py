import pickle
import torch
from torch.utils.data import Dataset
class MTCNNDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["image"], sample["label"], sample["roi"], sample.get("pts", torch.zeros(10)) 
