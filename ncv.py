from os import path, mkdir
from datetime import datetime
from timeit import default_timer

from torch import device, cuda
from scipy.stats import reciprocal
from sklearn.model_selection import KFold, ParameterSampler

from data import SubjectList, SegmentDataset
from model import DynamicCNN
from model import train_model, evaluate_model

# === SETTINGS ===
START_TIME = datetime.now()
DEVICE = device("cuda" if cuda.is_available() else "cpu")
OUTER_K, INNER_K = 5, 4
RANDOM_STATE = 10

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
        "ITERATIONS": 55,
        "GRID": {
            "LR": reciprocal(1e-5, 1e-4),
            "BATCH_SIZE": [16, 32, 64],
            "EPOCHS": [20, 35, 60, 100],
            "WEIGHT_DECAY": reciprocal(1e-5, 1e-3)
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
┃ TRAINING CONFIGURATION                                                  ┃
┃     OUTER FOLDS [{OUTER_K}]                                                     ┃
┃     INNER FOLDS [{INNER_K}]                                                     ┃
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

        # Create hyperparameter configurations + inner fold split for inner cross validation
        inner_folds = KFold(n_splits=inner_k, shuffle=True, random_state=i_outer)
        configs = list(ParameterSampler(hyperparameters["GRID"], hyperparameters["ITERATIONS"], random_state=i_outer))
        for i_config, config in enumerate(configs):
            
            inner_train_set = [subject_list[i] for i in i_outer_train]
            inner_val_set = [subject_list[i] for i in i_outer_test]



if __name__ == "__main__":
    destination = path.join(path.abspath("output"), f"data-{START_TIME.strftime('%Y%m%d-%H%M%S')}")
    mkdir(destination)

    OUTPUT_PATH = destination
    SUBJECT_LIST = SubjectList(path.abspath("data"))

    ncv(inner_k=INNER_K, outer_k=OUTER_K, network_type="DEBUG", hyperparameter_set="DEBUG", subject_list=SUBJECT_LIST, output_path=destination, random_state=RANDOM_STATE)