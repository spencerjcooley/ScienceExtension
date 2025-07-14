import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data import PatientDataset, SegmentDataset, SubjectData
from model import OSA_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTER_K = 5
INNER_K = 4

PATIENT_DATA = PatientDataset("data")

kf_outer = KFold(n_splits=OUTER_K, shuffle=True, random_state=1)
for outer_fold, (train_idx, test_idx) in enumerate(kf_outer.split(range(len(PATIENT_DATA)))):
    print(f"[OUTER {outer_fold+1}/{OUTER_K}]")

    # OUTER FOLDS
    outer_train_subjects = [PATIENT_DATA[i] for i in train_idx]
    outer_test_subjects = [PATIENT_DATA[i] for i in test_idx]

    # === INNER LOOP ===
    kf_inner = KFold(n_splits=INNER_K, shuffle=True, random_state=outer_fold)

    best_inner_loss = float('inf')
    best_model_state = None

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(kf_inner.split(outer_train_subjects)):
        print(f"    [INNER {inner_fold+1}/{INNER_K}]")

        inner_train_subjects = [outer_train_subjects[i] for i in inner_train_idx]
        inner_val_subjects = [outer_train_subjects[i] for i in inner_val_idx]

        inner_train_loader = DataLoader(SegmentDataset(inner_train_subjects), batch_size=32, shuffle=True)
        inner_val_loader = DataLoader(SegmentDataset(inner_val_subjects), batch_size=32)
        # Batch Size 32 is a placeholder, This will later use random search in a hyperparameter set