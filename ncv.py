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
    "CONV1": [15,32,1,2],
    "CONV2": [7,64,1,3],
    "CONV3": [5,128,1,2],
    "LINEAR": 64
}

SMALL_NETWORK = {
    "CONV1": [15,16,1,7],
    "CONV2": [7,32,1,3],
    "CONV3": [3,64,1,1],
    "LINEAR": 64
}

# PARAMETERS
PARAMETERS_DEBUG = {
    "ITERATIONS": 2,
    "GRID": {
        "LR": reciprocal(1e-5, 1e-2),
        "BATCH_SIZE": [128],
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



def ncv(OUTER_K: int, INNER_K: int, PARAMETERS_N: int, PARAMETERS_GRID: dict, SUBJECT_LIST: SubjectList, OUTPUT_PATH: str, RANDOM_STATE: int = 1, SMALL_MODEL: bool = False):
    print(
"""┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ OUTER ┃ CONFIGURATION ┃ INNER ┃  ELAPSED  ┃  EPOCH  ┃    LOSS    ┃
┗━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛""")

    outer_folds = KFold(n_splits=OUTER_K, shuffle=True, random_state=RANDOM_STATE)
    for i_OUTER, (i_outer_train, i_outer_test) in enumerate(outer_folds.split(SUBJECT_LIST.subjects)):
        print("┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓")
        print("┗━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛", end="\r")

        t_OUTER = default_timer()
        OUTPUT = {
            "train_indices": i_outer_train.tolist(),
            "test_indices": i_outer_test.tolist(),
            "summary": {},
            "configs": {}
        }

        outer_train_set = [SUBJECT_LIST[i] for i in i_outer_train]
        outer_test_set = [SUBJECT_LIST[i] for i in i_outer_test]

        best_inner_loss = float('inf')
        best_config = None
        i_best_config = None

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

            for i_INNER, (i_inner_train, i_inner_val) in enumerate(inner_folds.split(outer_train_set)):
                t_INNER = default_timer()

                inner_train_loader = DataLoader(SegmentDataset([outer_train_set[i] for i in i_inner_train]), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
                inner_val_loader = DataLoader(SegmentDataset([outer_train_set[i] for i in i_inner_val]), batch_size=VAL_BATCH_SIZE)

                if SMALL_MODEL: model = OSA_CNN(conv1_config=SMALL_NETWORK["CONV1"], conv2_config=SMALL_NETWORK["CONV2"], conv3_config=SMALL_NETWORK["CONV3"], linear_neurons=SMALL_NETWORK["LINEAR"], dropout=CONFIG["DROPOUT"]).to(device=DEVICE)
                else: model = OSA_CNN(conv1_config=NETWORK["CONV1"], conv2_config=NETWORK["CONV2"], conv3_config=NETWORK["CONV3"], linear_neurons=NETWORK["LINEAR"], dropout=CONFIG["DROPOUT"]).to(device=DEVICE)
                optimiser = optim.AdamW(params=model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

                epochs = train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"], dataloader=inner_train_loader, val_dataloader=inner_val_loader)
                loss = evaluate_model(model=model, device=DEVICE, dataloader=inner_val_loader)
                total_loss += loss
                elapsed_time = default_timer() - t_INNER

                OUTPUT["configs"][i_CONFIG+1]["inner_folds"][i_INNER+1] = {
                    "time": elapsed_time,
                    "epochs": epochs,
                    "loss": loss
                }

                if i_CONFIG == 0 and i_INNER == 0: print(f"┃  {i_OUTER+1}/{OUTER_K}  ┃     {i_CONFIG+1:02d}/{PARAMETERS_N:02d}     ┃  {i_INNER+1}/{INNER_K}  ┃  {f'{elapsed_time}'[:7]}  ┃   {epochs:03d}   ┃  {f'{loss}'[:8]}  ┃")
                elif i_INNER == 0: print(f"┃       ┃     {i_CONFIG+1:02d}/{PARAMETERS_N:02d}     ┃  {i_INNER+1}/{INNER_K}  ┃  {f'{elapsed_time}'[:7]}  ┃   {epochs:03d}   ┃  {f'{loss}'[:8]}  ┃")
                elif i_INNER != 0: print(f"┃       ┃       -       ┃  {i_INNER+1}/{INNER_K}  ┃  {f'{elapsed_time}'[:7]}  ┃   {epochs:03d}   ┃  {f'{loss}'[:8]}  ┃")
                print("┗━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛", end="\r")
            
            mean_loss = total_loss / INNER_K
            if mean_loss < best_inner_loss:
                best_inner_loss = mean_loss
                best_config = CONFIG
                i_best_config = i_CONFIG

            OUTPUT["configs"][i_CONFIG+1]["summary"] = {
                "time": default_timer() - t_CONFIG,
                "mean_loss": mean_loss
            }
        
        print("┣━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━━━┫")
        print("┗━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛", end="\r")
        
        t_TEST = default_timer()
        outer_train_loader = DataLoader(SegmentDataset(outer_train_set), batch_size=best_config["BATCH_SIZE"], shuffle=True)
        outer_test_loader = DataLoader(SegmentDataset(outer_test_set), batch_size=VAL_BATCH_SIZE, shuffle=False)

        if SMALL_MODEL: model = OSA_CNN(conv1_config=SMALL_NETWORK["CONV1"], conv2_config=SMALL_NETWORK["CONV2"], conv3_config=SMALL_NETWORK["CONV3"], linear_neurons=SMALL_NETWORK["LINEAR"], dropout=best_config["DROPOUT"]).to(device=DEVICE)
        else: model = OSA_CNN(conv1_config=NETWORK["CONV1"], conv2_config=NETWORK["CONV2"], conv3_config=NETWORK["CONV3"], linear_neurons=NETWORK["LINEAR"], dropout=best_config["DROPOUT"]).to(device=DEVICE)
        optimiser = optim.AdamW(params=model.parameters(), lr=best_config["LR"], weight_decay=best_config["WEIGHT_DECAY"])

        epochs = train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=best_config["EPOCHS"], patience=best_config["PATIENCE"], dataloader=outer_train_loader)
        performance = evaluate_model_full(model=model, device=DEVICE, dataloader=outer_test_loader, threshold=0.5)
        outer_loss = performance['loss']

        OUTPUT["summary"] = {
            "network": SMALL_NETWORK if SMALL_MODEL else NETWORK,
            "time": default_timer() - t_OUTER,
            "best_config": best_config,
            "performance": performance
        }

        with open(os.path.join(OUTPUT_PATH, f"MODEL{i_OUTER+1}.json"), "w") as file: json.dump(OUTPUT, file, indent=4)

        print(f"┃  {i_OUTER+1}/{OUTER_K}  ┃     {i_best_config+1:02d}/{PARAMETERS_N:02d}     ┃   -   ┃  {f'{default_timer() - t_TEST}'[:7]}  ┃   {epochs:03d}   ┃  {f'{outer_loss}'[:8]}  ┃")
        print("┣━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━━━┫")
        print(f"┃ TOTAL ┃       -       ┃   -   ┃  {f'{default_timer() - t_OUTER}'[:7]}  ┃    -    ┃     --     ┃")
        print("┗━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━┛")



if __name__ == "__main__":
    if not os.path.exists("output"): os.mkdir("output")
    destination = os.path.join(os.path.abspath("output"), f"data-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.mkdir(destination)

    OUTPUT_PATH = destination
    SUBJECT_LIST = SubjectList(os.path.abspath("data"))
    ncv(OUTER_K, INNER_K, PARAMETERS["ITERATIONS"], PARAMETERS["GRID"], SUBJECT_LIST, OUTPUT_PATH, RANDOM_STATE=10, SMALL_MODEL=True)