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
    "BATCH_SIZE": [8, 64],
    "EPOCHS": [1]
}

# PARAMETER_ITER = 10
# PARAMETER_GRID = {
#     "LR": reciprocal(1e-5, 1e-2),
#     "BATCH_SIZE": [8, 16, 32, 64],
#     "EPOCHS": [50, 100, 150, 200]
# }

PATIENT_DATA = PatientDataset("data")

kf_outer = KFold(n_splits=OUTER_K, shuffle=True, random_state=1)
results = []
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
    for i_parameter_config, parameter_config in enumerate(parameter_configs):
        print(f"    [CONFIG {i_parameter_config+1}/{PARAMETER_ITER}] | [LR: {parameter_config['LR']:.4f} | B: {parameter_config['BATCH_SIZE']:02d} | E: {parameter_config['EPOCHS']:03d}]")
        fold_losses = []

        for i_inner_fold, (inner_train_idx, inner_val_idx) in enumerate(kf_inner.split(outer_train_subjects)):
            T_0 = default_timer()
            print(f"        [INNER {i_inner_fold+1}/{INNER_K}] | ", end="")

            inner_train_subjects = [outer_train_subjects[i] for i in inner_train_idx]
            inner_val_subjects = [outer_train_subjects[i] for i in inner_val_idx]

            inner_train_loader = DataLoader(SegmentDataset(inner_train_subjects), batch_size=parameter_config["BATCH_SIZE"], shuffle=True)
            inner_val_loader = DataLoader(SegmentDataset(inner_val_subjects), batch_size=parameter_config["BATCH_SIZE"])

            model = OSA_CNN().to(DEVICE)
            optimiser = torch.optim.Adam(model.parameters(), lr=parameter_config["LR"])

            train_model(model=model, dataloader=inner_train_loader, epochs=parameter_config["EPOCHS"], optimiser=optimiser, device=DEVICE)
            val_loss = evaluate_model(model=model, dataloader=inner_val_loader, device=DEVICE)
            fold_losses.append(val_loss)

            print(f"L: {val_loss:.4f} | T: {f'{default_timer() - T_0:.5f}'[:6]}s")

        mean_loss = sum(fold_losses) / len(fold_losses)
        print(f"    [CONFIG {i_parameter_config+1}/{PARAMETER_ITER}] | Avg L: {mean_loss:.4f}")

        if mean_loss < best_inner_loss:
            best_inner_loss = mean_loss
            best_model_config = parameter_config
    
    print(f"[OUTER {i_outer_fold+1}/{OUTER_K}] | ...", end="\r")
    outer_train_loader = DataLoader(SegmentDataset(outer_train_subjects), batch_size=best_model_config["BATCH_SIZE"], shuffle=True)
    model = OSA_CNN().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=best_model_config["LR"])

    train_model(model=model, dataloader=outer_train_loader, epochs=best_model_config["EPOCHS"], optimiser=optimiser, device=DEVICE)

    outer_test_loader = DataLoader(SegmentDataset(outer_test_subjects), batch_size=best_model_config["BATCH_SIZE"])
    outer_test_metrics = evaluate_model_full(model=model, dataloader=outer_test_loader, device=DEVICE)
    print(f"[OUTER {i_outer_fold+1}/{OUTER_K}] | L: {outer_test_metrics['loss']:.4f} | A: {outer_test_metrics['accuracy']:.4f}\n")

    results.append({
        "fold": i_outer_fold,
        **outer_test_metrics
    })

# TEMPORARY RESULT DEBUG
for result in results: print(result)