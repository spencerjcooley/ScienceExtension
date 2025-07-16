from os import path, mkdir
from datetime import datetime

from torch import device, cuda
from scipy.stats import reciprocal
from sklearn.model_selection import KFold

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
        "ITERATIONS": 10,
            "ITERATIONS": 15,
        "GRID": {
            "LR": reciprocal(1e-5, 1e-4),
            "BATCH_SIZE": [16, 32, 64],
            "EPOCHS": [50, 100, 150],
            "PATIENCE": [5, 10, 15],
            "WEIGHT_DECAY": reciprocal(1e-5, 1e-3)
        }
    },
    "DEBUG": {
        "ITERATIONS": 2,
        "GRID": {
            "LR": reciprocal(1e-4, 1e-3),
            "BATCH_SIZE": [128],
            "EPOCHS": [5],
            "PATIENCE": [1, 2, 3],
            "WEIGHT_DECAY": reciprocal(1e-4, 1e-2)
        }
    }
}



# === NESTED CROSS VALIDATION ===
def ncv(inner_k: int, outer_k: int, network_type: str, hyperparameter_set: dict, subject_list: SubjectList, output_path: str, random_state: int = 1):
    NETWORK = NETWORKS[network_type]

    # === OVERVIEW HEADER ===
    print(f"""┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ NCV TEST {START_TIME.strftime('[%Y-%m-%d] [%H:%M:%S]')}                           ┃
┃                                                            ┃
┃ TRAINING CONFIGURATION                                     ┃
┃     OUTER FOLDS [{OUTER_K}]                                        ┃
┃     INNER FOLDS [{INNER_K}]                                        ┃
┃                                                            ┃
┃ MODEL CONFIGURATION [{network_type}]{' '*(37-len(network_type))}┃""")
    
    for i, layer in enumerate(NETWORK):
        if i == 0: print(f"┃     {layer['type'].upper()}{' '*(55-len(layer['type']))}┃")
        else: print(f"┃     → {layer['type'].upper()}{' '*(53-len(layer['type']))}┃")

    print("""┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ OUTER ┃ CONFIGS ┃ INNER ┃  ELAPSED  ┃  EPOCH  ┃    LOSS    ┃
┗━━━━━━━┻━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛""")



if __name__ == "__main__":
    destination = path.join(path.abspath("output"), f"data-{START_TIME.strftime('%Y%m%d-%H%M%S')}")
    mkdir(destination)

    OUTPUT_PATH = destination
    SUBJECT_LIST = SubjectList(path.abspath("data"))

    ncv(inner_k=INNER_K, outer_k=OUTER_K, network_type="MAIN", hyperparameter_set=HYPERPARAMETERS["DEBUG"], subject_list=SUBJECT_LIST, output_path=destination, random_state=RANDOM_STATE)