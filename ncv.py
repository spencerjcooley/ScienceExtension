import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterSampler
from scipy.stats import reciprocal
from timeit import default_timer

from data import PatientDataset, SegmentDataset
from model import OSA_CNN
from train import train_model, evaluate_model, evaluate_model_full

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTER_K = 5
INNER_K = 4

# TEMPORARY VALUES FOR FAST TESTING
PARAMETER_ITER = 1
PARAMETER_GRID = {
    "LR": reciprocal(1e-5, 1e-2),
    "BATCH_SIZE": [32, 64, 128],
    "EPOCHS": [1],
    "PATIENCE": [5, 10, 15],
    "DROPOUT": [0.3, 0.5, 0.7],
    "WEIGHT_DECAY": reciprocal(1e-4, 1e-2)
}

# PARAMETER_ITER = 20
# PARAMETER_GRID = {
#     "LR": reciprocal(1e-5, 1e-2),
#     "BATCH_SIZE": [16, 32, 64, 128],
#     "EPOCHS": [50, 100, 150, 200],
#     "PATIENCE": [5, 10, 15],
#     "DROPOUT": [0.3, 0.5, 0.7],
#     "WEIGHT_DECAY": reciprocal(1e-4, 1e-2)
# }

PATIENT_DATA = PatientDataset("data")

results = []
kf_outer = KFold(n_splits=OUTER_K, shuffle=True, random_state=1)
for i_outer_fold, (train_idx, test_idx) in enumerate(kf_outer.split(range(len(PATIENT_DATA)))):
    print(f"[OUTER {i_outer_fold+1}/{OUTER_K}]")

    # OUTER FOLDS
    outer_train_subjects = [PATIENT_DATA[i] for i in train_idx]
    outer_test_subjects = [PATIENT_DATA[i] for i in test_idx]

    # === INNER LOOP ===
    kf_inner = KFold(n_splits=INNER_K, shuffle=True, random_state=i_outer_fold)

    best_inner_loss = float('inf')
    best_model_config = None

    parameter_configs = list(ParameterSampler(PARAMETER_GRID, n_iter=PARAMETER_ITER, random_state=i_outer_fold))
    for i_parameter_config, config in enumerate(parameter_configs):
        print(f"    [CONFIG {i_parameter_config+1}/{PARAMETER_ITER}] | [LR: {config['LR']:.4f} | BS: {config['BATCH_SIZE']:02d} | E: {config['EPOCHS']:03d} | P: {config['PATIENCE']:02d} | D: {config['DROPOUT']} | WD: {config['WEIGHT_DECAY']:.4f}]")
        fold_losses = []

        for i_inner_fold, (inner_train_idx, inner_val_idx) in enumerate(kf_inner.split(outer_train_subjects)):
            T_0 = default_timer()
            print(f"        [INNER {i_inner_fold+1}/{INNER_K}] | ", end="")

            inner_train_subjects = [outer_train_subjects[i] for i in inner_train_idx]
            inner_val_subjects = [outer_train_subjects[i] for i in inner_val_idx]

            inner_train_loader = DataLoader(SegmentDataset(inner_train_subjects), batch_size=config["BATCH_SIZE"], shuffle=True)
            inner_val_loader = DataLoader(SegmentDataset(inner_val_subjects), batch_size=config["BATCH_SIZE"])

            model = OSA_CNN().to(DEVICE)
            optimiser = torch.optim.AdamW(params=model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

            train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=config["EPOCHS"], patience=config["PATIENCE"], dataloader=inner_train_loader, val_dataloader=inner_val_loader)
            val_loss = evaluate_model(model=model, device=DEVICE, dataloader=inner_val_loader)
            fold_losses.append(val_loss)

            print(f"L: {val_loss:.4f} | T: {f'{default_timer() - T_0:.5f}'[:6]}s")

        mean_loss = sum(fold_losses) / len(fold_losses)
        print(f"    [CONFIG {i_parameter_config+1}/{PARAMETER_ITER}] | Avg L: {mean_loss:.4f}")

        if mean_loss < best_inner_loss:
            best_inner_loss = mean_loss
            best_model_config = config
    
    print(f"[OUTER {i_outer_fold+1}/{OUTER_K}] | ...", end="\r")
    outer_train_loader = DataLoader(SegmentDataset(outer_train_subjects), batch_size=best_model_config["BATCH_SIZE"], shuffle=True)
    model = OSA_CNN().to(DEVICE)
    optimiser = torch.optim.AdamW(params=model.parameters(), lr=best_model_config["LR"], weight_decay=best_model_config["WEIGHT_DECAY"])

    train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=best_model_config["EPOCHS"], patience=0, dataloader=outer_train_loader, val_dataloader=None)

    outer_test_loader = DataLoader(SegmentDataset(outer_test_subjects), batch_size=best_model_config["BATCH_SIZE"])
    outer_test_metrics = evaluate_model_full(model=model, device=DEVICE, dataloader=outer_test_loader, threshold=0.5)
    print(f"[OUTER {i_outer_fold+1}/{OUTER_K}] | L: {outer_test_metrics['loss']:.4f} | A: {outer_test_metrics['accuracy']:.4f}\n")

    results.append({
        "fold": i_outer_fold,
        **outer_test_metrics
    })