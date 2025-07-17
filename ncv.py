from os import path, mkdir
from math import log, exp, prod
from datetime import datetime
from timeit import default_timer

from torch import device, cuda
from torch.utils.data import DataLoader
from torch.optim import AdamW
from scipy.stats import reciprocal
from sklearn.model_selection import KFold, ParameterSampler

from data import SubjectList, SegmentDataset
from model import DynamicCNN
from model import train_model, evaluate_model



# === TOOLS ===
def insert_logarithmic_means(start: float, end: float, n_means: int, is_int: bool = True):
    d = (log(end) - log(start)) / (n_means + 1)
    return [round(exp(log(start) + i * d)) for i in range(n_means + 2)] if is_int else [exp(log(start) + i * d) for i in range(n_means + 2)]

def insert_arithmetic_means(start: int, end: int, n_means: int, is_int: bool = True):
    d = (end - start)/(n_means + 1)
    return [round(start + i * d) for i in range(n_means + 2)] if is_int else [start + i * d for i in range(n_means + 2)]



# === SETTINGS ===
START_TIME = datetime.now()

OUTPUT_PATH = path.join(path.abspath("output"), f"data-{START_TIME.strftime('%Y%m%d-%H%M%S')}")
SUBJECT_LIST = SubjectList(path.abspath("data"))

DEVICE = device("cuda" if cuda.is_available() else "cpu")
OUTER_K, INNER_K = 5, 4
RANDOM_STATE = 10

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 95 # < 100

NETWORKS = {
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
    "BASIC": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
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

HYPERPARAMETERS = {
    "MAIN": {
        "ITERATIONS": round(log(ALPHA)/log((TARGET_PERCENTILE/100))), # Iteration estimation via X~Bin(n,p)
        "GRID": {
            "LR": insert_logarithmic_means(start=1e-5, end=1e-4, n_means=3, is_int=False),
            "BATCH_SIZE": insert_logarithmic_means(start=16, end=128, n_means=2),
            "EPOCHS": insert_logarithmic_means(start=50, end=150, n_means=3),
            "WEIGHT_DECAY": insert_logarithmic_means(start=1e-5, end=1e-3, n_means=3, is_int=False)
        }
    },
    "DEBUG": {
        "ITERATIONS": 2,
        "GRID": {
            "LR": reciprocal(1e-5, 1e-4),
            "BATCH_SIZE": [128],
            "EPOCHS": [5],
            "WEIGHT_DECAY": reciprocal(1e-5, 1e-3)
        }
    }
}



# === NESTED CROSS VALIDATION ===
def ncv(inner_k: int, outer_k: int, network_type: str, hyperparameter_set: str, subject_list: SubjectList, output_path: str, random_state: int = 1):
    t_ncv = default_timer()
    network, hyperparameters = NETWORKS[network_type], HYPERPARAMETERS[hyperparameter_set]

    # === OVERVIEW HEADER ===
    print(f"""┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ NCV TEST {START_TIME.strftime('[%Y-%m-%d] [%H:%M:%S]')}                                        ┃
┃                                                                         ┃
┃ TRAINING SETTINGS.                                                      ┃
┃     OUTER FOLDS [{OUTER_K}]                                                     ┃
┃     INNER FOLDS [{INNER_K}]                                                     ┃
┃                                                                         ┃
┃ RANDOM SEARCH SETTINGS                                                  ┃
┃     CONFIDENCE [{int((1-ALPHA)*100):02d}%]                                                    ┃
┃     PERCENTILE [{TARGET_PERCENTILE:02d}th]                                                   ┃
┃     ITERATIONS [{hyperparameters['ITERATIONS']:02d}]                                                     ┃
┃     COVERAGE [{hyperparameters['ITERATIONS']*100//prod([len(options) for options in hyperparameters["GRID"].values()]):02d}%]                                                      ┃
┃                                                                         ┃
┃ MODEL CONFIGURATION [{network_type}]{' '*(50-len(network_type))}┃""")
    
    for i, layer in enumerate(network):
        if i == 0: print(f"┃     {layer['type'].upper()}{' '*(68-len(layer['type']))}┃")
        else: print(f"┃     → {layer['type'].upper()}{' '*(66-len(layer['type']))}┃")

    print("""┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ OUTER ┃ CONFIGS ┃ INNER ┃  ELAPSED  ┃  EPOCH  ┃    LOSS    ┃     F1     ┃
┗━━━━━━━┻━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━┛""")
    


    # === OUTER FOLDS ===

    # Split data patient wise into K folds for outer CV loop
    outer_folds = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    for i_outer, (i_outer_train, i_outer_test) in enumerate(outer_folds.split(subject_list.subjects)):
        t_outer = default_timer()

        # Create list of custom SubjectData objects (in data/ecg_dataset.py)
        outer_train_set = [subject_list[i] for i in i_outer_train]
        outer_test_set = [subject_list[i] for i in i_outer_test]

        # Tracking best hyperparameter settings
        best_f1 = 0
        best_config = None

        # Create hyperparameter configurations + inner fold split for inner cross validation
        inner_folds = KFold(n_splits=inner_k, shuffle=True, random_state=i_outer)
        configs = list(ParameterSampler(hyperparameters["GRID"], hyperparameters["ITERATIONS"], random_state=i_outer))
        for i_config, config in enumerate(configs):
            t_config = default_timer()

            for i_inner, (i_inner_train, i_inner_test) in enumerate(inner_folds.split(outer_train_set)):
                t_inner = default_timer()

                # Create inner DataLoaders for training/testing
                inner_train_loader = DataLoader(dataset=SegmentDataset([subject_list[i] for i in i_inner_train]), batch_size=config["BATCH_SIZE"], shuffle=True)
                inner_test_loader = DataLoader(dataset=SegmentDataset([subject_list[i] for i in i_inner_test]), batch_size=config["BATCH_SIZE"])

                model = DynamicCNN(network)
                optimiser = AdamW(params=model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

                train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=config["EPOCHS"], dataloader=inner_train_loader)
                performance = evaluate_model(model=model, device=DEVICE, dataloader=inner_test_loader)



        # FOR JSON SAVING
        # mkdir(output_path)



if __name__ == "__main__":
    ncv(inner_k=INNER_K, outer_k=OUTER_K, network_type="MAIN", hyperparameter_set="MAIN", subject_list=SUBJECT_LIST, output_path=OUTPUT_PATH, random_state=RANDOM_STATE)