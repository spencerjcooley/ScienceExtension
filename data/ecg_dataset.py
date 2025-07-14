import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith('npz')]
        self.segments, self.labels = [], []
        
        for file_path in self.file_paths:
            data = np.load(file_path)
            
            segments = data['segments'] # shape: (N, 6000)
            labels = data['labels']     # shape: (N,)

            self.segments.append(segments)
            self.labels.append(labels)

        self.segments = np.concatenate(self.segments, axis=0) # (N, 6000)
        self.labels = np.concatenate(self.labels, axis=0)

        self.segments = torch.tensor(self.segments, dtype=torch.float32).unsqueeze(1) # Add Channel Dimension | (N, 1, 6000)
        self.labels = torch.tensor(self.labels, dtype=torch.float32) # (N, 1)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.segments[idx], self.labels[idx]