# import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data import PatientDataset, SegmentDataset, SubjectData
from model import OSA_CNN

PATIENT_DATA = PatientDataset("data")

kf_outer = KFold(n_splits=5)
for train_idx, test_idx in kf_outer.split(range(len(PATIENT_DATA))):
    # Get Subjects
    train_subjects = [PATIENT_DATA[i] for i in train_idx]
    test_subjects = [PATIENT_DATA[i] for i in test_idx]

    # Create Merged Datasets + Loaders for Pytorch
    train_dataset = SegmentDataset(train_subjects)
    test_dataset = SegmentDataset(test_subjects)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)