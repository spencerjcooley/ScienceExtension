import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SubjectData:
    def __init__(self, segments, labels):
        if isinstance(segments, torch.Tensor):
            self.x = segments.detach().clone().float()
            if self.x.dim() == 2: self.x = self.x.unsqueeze(1)
        else: self.x = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
        if isinstance(labels, torch.Tensor): self.y = labels.detach().clone().float()
        else: self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.y)
    
class SubjectList:
    def __init__(self, data_dir: str):
        self.subjects = []
        npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        for file_path in npz_files:
            print(file_path)
            if not file_path.endswith('.npz'): continue
            data = np.load(os.path.join(data_dir, file_path))
            self.subjects.append(SubjectData(segments=data['segments'], labels=data['labels']))

    def __len__(self): return len(self.subjects)
    def __getitem__(self, i): return self.subjects[i]

class SegmentDataset(Dataset):
    def __init__(self, subject_list: list[SubjectData]):
        self.data = torch.cat([subject.x for subject in subject_list], dim=0)
        self.labels = torch.cat([subject.y for subject in subject_list], dim=0)
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]