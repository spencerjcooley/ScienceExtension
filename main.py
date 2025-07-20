import re
import json
import numpy as np
from os import path
from math import log, exp
from datetime import datetime
from timeit import default_timer
from warnings import filterwarnings
import matplotlib.pyplot as plt

from torch import cuda
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, ParameterSampler
filterwarnings("ignore", category=UserWarning, module='sklearn\.model_selection\..*')

from data import SubjectList, SegmentDataset
from model import DynamicCNN, train_model, evaluate_model, FocalLoss


# === TOOLS ===
def insert_logarithmic_means(start: float, end: float, n_means: int, is_int: bool = True):
    d = (log(end) - log(start)) / (n_means + 1)
    return [round(exp(log(start) + i * d)) for i in range(n_means + 2)] if is_int else [exp(log(start) + i * d) for i in range(n_means + 2)]

def insert_arithmetic_means(start: int, end: int, n_means: int, is_int: bool = True):
    d = (end - start)/(n_means + 1)
    return [round(start + i * d) for i in range(n_means + 2)] if is_int else [start + i * d for i in range(n_means + 2)]

def stratified_subject_split(subject_list: SubjectList, n_splits: int = 5, seed: int = 42):
    subjects = subject_list
    n_subjects = len(subjects)
    prevalence = np.array([subject.y.mean().item() for subject in subjects])
    bins = np.quantile(prevalence, [0.2, 0.4, 0.6, 0.8])
    binned_labels = np.digitize(prevalence, bins=bins)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(np.arange(n_subjects), binned_labels):
        train_subjects = [subjects[i] for i in train_idx]
        val_subjects = [subjects[i] for i in val_idx]
        folds.append((train_subjects, val_subjects))

    return folds

replace_func = lambda match: " ".join(match.group().split())



# === SETTINGS ===
START_TIME = datetime.now()

OUTPUT_PATH = path.join(path.abspath("output"), f"data-{START_TIME.strftime('%Y%m%d-%H%M%S')}")
SUBJECT_LIST = SubjectList(path.abspath("data"))

DEVICE = "cuda" if cuda.is_available() else "cpu"
OUTER_K, INNER_K = 5, 4
TEST_BATCH_SIZE = 512

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 90 # < 100

MODELS = {
    "MAIN": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "conv1d", "in_channels": 32, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 64},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 64, "out_features": 1}
    ],
    "STRIDED": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 32, "kernel_size": 8, "stride": 2, "padding": 2},
        {"type": "gelu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "conv1d", "in_channels": 32, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "gelu"},
        {"type": "batchnorm1d", "num_features": 64},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "in_features": 64, "out_features": 1}
    ],
    "STRIDED_V2": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "dropout", "p": 0.5},
        {"type": "linear", "in_features": 32, "out_features": 1}
    ],
    "BASIC": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 32, "out_features": 1}
    ],
    "DEBUG": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 16, "out_features": 1}
    ]
}

NCV_CONFIGS = {
    "MAIN": {
        "ITERATIONS": int(-(-log(ALPHA) // log((TARGET_PERCENTILE/100)))), # Iteration estimation via X~Bin(n,p) | Ceiling Function
        "GRID": {
            "LR": insert_logarithmic_means(start=5e-4, end=5e-3, n_means=3, is_int=False),
            "BATCH_SIZE": insert_logarithmic_means(start=32, end=128, n_means=1),
            "EPOCHS": insert_logarithmic_means(start=50, end=100, n_means=2),
            "ALPHA": insert_arithmetic_means(0.5, 0.95, n_means=2),
            "GAMMA": insert_arithmetic_means(1, 3, n_means=1)
        }
    },
    "DEBUG": {
        "ITERATIONS": 3,
        "GRID": {
            "LR": [1e-3],
            "BATCH_SIZE": [32],
            "EPOCHS": [100],
            "ALPHA": [0.85],
            "GAMMA": [2.0]
        }
    }
}



def cv(k: int, network: dict, config: dict, test_batch_size: int, subject_list: list, random_seed: int):
    t = default_timer()
    output = { "summary": {} }

    f1_scores = []
    loss_scores = []

    sfk = stratified_subject_split(subject_list, k, seed=random_seed)
    loss_function = FocalLoss(config["ALPHA"], config["GAMMA"])
    for i, (train_list, test_list) in enumerate(sfk, 1):
        t_fold = default_timer()

        train_loader = DataLoader(SegmentDataset(train_list), config["BATCH_SIZE"], shuffle=True)
        test_loader = DataLoader(SegmentDataset(test_list), test_batch_size, shuffle=True)

        model = DynamicCNN(network).to(DEVICE)
        optimiser = AdamW(model.parameters(), lr=config["LR"])
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=config["LR"], steps_per_epoch=len(train_loader), epochs=config["EPOCHS"])

        losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, config["EPOCHS"], train_loader)
        performance_train = evaluate_model(model, DEVICE, loss_function, train_loader)
        performance_test = evaluate_model(model, DEVICE, loss_function, test_loader)

        output[i] = {
            "time": default_timer() - t_fold,
            "losses": losses,
            "train_perf": performance_train,
            "test_perf": performance_test
        }

        f1_scores.append(performance_test["metrics"]["f1"])
        loss_scores.append(performance_test["loss"])

    output["summary"] = {
        "time": default_timer() - t,
        "mean_loss": np.mean(loss_scores),
        "mean_f1": np.mean(f1_scores)
    }

    return output



def ncv(outer_k: int, inner_k: int, network: dict, hyperparameters: dict, test_batch_size: int, subject_list: SubjectList, random_seed: int = 42):
    sfk = stratified_subject_split(subject_list.subjects, outer_k, random_seed)
    for i_outer, (train_list, test_list) in enumerate(sfk, 1):
        configs = list(ParameterSampler(hyperparameters["GRID"], hyperparameters["ITERATIONS"], random_state=i_outer))
        for i_config, config in enumerate(configs, 1):
            config_data = cv(inner_k, network, config, test_batch_size, train_list, random_seed=(i_outer*i_config))

if __name__ == "__main__":
    ncv(OUTER_K, INNER_K, MODELS["STRIDED_V2"], NCV_CONFIGS["DEBUG"], TEST_BATCH_SIZE, SUBJECT_LIST, 42)