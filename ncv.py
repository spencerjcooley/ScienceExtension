import os
import json
from timeit import default_timer
from datetime import datetime

from train import train_model, evaluate_model, evaluate_model_full
from data import SubjectList, SegmentDataset
from model import OSA_CNN

from torch import device, cuda, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterSampler
from scipy.stats import reciprocal

DEVICE = device("cuda" if cuda.is_available() else "cpu")
OUTER_K, INNER_K = 5, 4
VAL_BATCH_SIZE = 256

# MAGIC NUMBERS
NETWORK = {
    "CONV1": (15,32,1,2),
    "CONV2": (7,64,1,3),
    "CONV3": (5,128,1,2),
    "LINEAR": 64
}

SMALL_NETWORK = {
    "CONV1": (15,16,1,7),
    "CONV2": (7,32,1,3),
    "CONV3": (3,64,1,1),
    "LINEAR": 64
}

# PARAMETERS
PARAMETERS_DEBUG = {
    "ITERATIONS": 1,
    "GRID": {
        "LR": reciprocal(1e-5, 1e-2),
        "BATCH_SIZE": [32, 64, 128],
        "EPOCHS": [1],
        "PATIENCE": [5, 10, 15],
        "DROPOUT": [0.3, 0.5, 0.7],
        "WEIGHT_DECAY": reciprocal(1e-4, 1e-2)
    }
}

PARAMETERS = {
    "ITERATIONS": 10,
    "GRID": {
        "LR": reciprocal(1e-5, 1e-2),
        "BATCH_SIZE": [16, 32, 64, 128],
        "EPOCHS": [50, 100, 150],
        "PATIENCE": [5, 10, 15],
        "DROPOUT": [0.3, 0.5, 0.7],
        "WEIGHT_DECAY": reciprocal(1e-4, 1e-2)
    }
}



def ncv(OUTER_K: int, INNER_K: int, PARAMETERS_N: int, PARAMETERS_GRID: dict, SUBJECT_LIST: SubjectList, OUTPUT_PATH: str, RANDOM_STATE: int = 1):
    outer_folds = KFold(n_splits=OUTER_K, shuffle=True, random_state=RANDOM_STATE)
    for i_OUTER, (i_outer_train, i_outer_test) in enumerate(outer_folds.split(SUBJECT_LIST.subjects)):
        t_OUTER = default_timer()
        OUTPUT = {
            "train_indices": i_outer_train,
            "test_indices": i_outer_test,
            "summary": {},
            "configs": {}
        }

        outer_train_set = [SUBJECT_LIST[i] for i in i_outer_train]
        outer_test_set = [SUBJECT_LIST[i] for i in i_outer_test]

        inner_folds = KFold(n_splits=INNER_K, shuffle=True, random_state=i_OUTER)
        CONFIGS = list(ParameterSampler(param_distributions=PARAMETERS_GRID, n_iter=PARAMETERS_N, random_state=i_OUTER))
        for i_CONFIG, CONFIG in enumerate(CONFIGS):
            t_CONFIG = default_timer()

            OUTPUT["configs"][i_CONFIG+1] = {
                "summary": {},
                "hyperparameters": CONFIG,
                "inner_folds": {}
            }

            total_loss = 0
            best_inner_loss = float('inf')
            best_config = None

            for i_INNER, (i_inner_train, i_inner_val) in enumerate(inner_folds.split(outer_train_set)):
                t_INNER = default_timer()

                inner_train_loader = DataLoader(SegmentDataset([outer_train_set[i] for i in i_inner_train]), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
                inner_val_loader = DataLoader(SegmentDataset([outer_train_set[i] for i in i_inner_val]), batch_size=VAL_BATCH_SIZE)

                model = OSA_CNN(conv1_config=NETWORK["CONV1"], conv2_config=NETWORK["CONV2"], conv3_config=NETWORK["CONV3"], linear_neurons=NETWORK["LINEAR"], dropout=CONFIG["DROPOUT"]).to(device=DEVICE)
                optimiser = optim.AdamW(params=model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

                epochs = train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"], dataloader=inner_train_loader, val_dataloader=inner_val_loader)
                loss = evaluate_model(model=model, device=DEVICE, dataloader=inner_val_loader, threshold=0.5)
                total_loss += loss

                OUTPUT["configs"][i_CONFIG+1]["inner_folds"][i_INNER+1] = {
                    "time": default_timer() - t_INNER,
                    "epochs": epochs,
                    "loss": loss
                }

            mean_loss = total_loss / INNER_K
            if mean_loss < best_inner_loss:
                best_inner_loss = mean_loss
                best_config = CONFIG

            OUTPUT["configs"][i_CONFIG+1]["summary"] = {
                "time": default_timer() - t_CONFIG,
                "mean_loss": mean_loss
            }
        
        outer_train_loader = DataLoader(SegmentDataset(outer_train_set), batch_size=best_config["BATCH_SIZE"], shuffle=True)
        outer_test_loader = DataLoader(SegmentDataset(outer_test_set), batch_size=VAL_BATCH_SIZE, shuffle=False)

        model = OSA_CNN(conv1_config=NETWORK["CONV1"], conv2_config=NETWORK["CONV2"], conv3_config=NETWORK["CONV3"], linear_neurons=NETWORK["LINEAR"], dropout=best_config["DROPOUT"])
        optimiser = optim.AdamW(params=model.parameters(), lr=best_config["LR"], weight_decay=best_config["WEIGHT_DECAY"])

        train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=best_config["EPOCHS"], patience=best_config["PATIENCE"], dataloader=outer_train_loader)
        performance = evaluate_model_full(model=model, device=DEVICE, dataloader=outer_test_loader, threshold=0.5)

        OUTPUT["summary"] = {
            "time": default_timer() - t_OUTER,
            "best_config": best_config,
            "performance": performance
        }

        with open(os.path.join(OUTPUT_PATH, f"MODEL{i_OUTER+1}.json"), "w") as file: json.dump(OUTPUT, file, indent=4)



if __name__ == "__main__":
    if not os.path.exists("output"): os.mkdir("output")
    destination = os.path.join(os.path.abspath("output"), f"data-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.mkdir(destination)

    OUTPUT_PATH = destination
    SUBJECT_LIST = SubjectList(os.path.abspath("data"))
    ncv(OUTER_K, INNER_K, PARAMETERS_DEBUG["ITERATIONS"], PARAMETERS_DEBUG["GRID"], SUBJECT_LIST, OUTPUT_PATH)