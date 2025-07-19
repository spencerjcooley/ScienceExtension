from re import sub
from json import dumps
from os import path, mkdir
from numpy import mean, std
from datetime import datetime
from math import log, exp, prod
from timeit import default_timer
from sys import argv
import json

from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterSampler

from data import SubjectList, SegmentDataset
from model import DynamicCNN, train_model, evaluate_model



# === TOOLS ===
def insert_logarithmic_means(start: float, end: float, n_means: int, is_int: bool = True):
    d = (log(end) - log(start)) / (n_means + 1)
    return [round(exp(log(start) + i * d)) for i in range(n_means + 2)] if is_int else [exp(log(start) + i * d) for i in range(n_means + 2)]

def insert_arithmetic_means(start: int, end: int, n_means: int, is_int: bool = True):
    d = (end - start)/(n_means + 1)
    return [round(start + i * d) for i in range(n_means + 2)] if is_int else [start + i * d for i in range(n_means + 2)]

replace_func = lambda match: " ".join(match.group().split())



# === SETTINGS ===
START_TIME = datetime.now()

OUTPUT_PATH = path.join(path.abspath("output"), f"data-{START_TIME.strftime('%Y%m%d-%H%M%S')}")
SUBJECT_LIST = SubjectList(path.abspath("data"))

DEVICE = "cuda" if cuda.is_available() else "cpu"
OUTER_K, INNER_K = 5, 4
RANDOM_STATE = 10
TEST_BATCH_SIZE = 256

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 90 # < 100

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
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 7, "stride": 1, "padding": 3},
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

HYPERPARAMETERS = {
    "MAIN": {
        "ITERATIONS": int(-(-log(ALPHA) // log((TARGET_PERCENTILE/100)))), # Iteration estimation via X~Bin(n,p) | Ceiling Function
        "GRID": {
            "LR": insert_logarithmic_means(start=1e-5, end=1e-4, n_means=3, is_int=False),
            "BATCH_SIZE": insert_logarithmic_means(start=32, end=128, n_means=1),
            "EPOCHS": insert_logarithmic_means(start=50, end=100, n_means=2),
            "WEIGHT_DECAY": insert_logarithmic_means(start=1e-5, end=1e-3, n_means=3, is_int=False)
        }
    },
    "DEBUG": {
        "ITERATIONS": 3,
        "GRID": {
            "LR": insert_logarithmic_means(start=1e-5, end=1e-4, n_means=3, is_int=False),
            "BATCH_SIZE": [128],
            "EPOCHS": [3],
            "WEIGHT_DECAY": insert_logarithmic_means(start=1e-5, end=1e-3, n_means=3, is_int=False)
        }
    }
}



# === NESTED CROSS VALIDATION ===
def ncv(inner_k: int, outer_k: int, network_type: str, hyperparameter_set: str, subject_list: SubjectList, output_path: str, random_state: int = 1):
    t_ncv = default_timer()
    network, hyperparameters = NETWORKS[network_type], HYPERPARAMETERS[hyperparameter_set]

    # === OVERVIEW HEADER ===
    print(f"""┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ NCV TEST {START_TIME.strftime('[%Y-%m-%d] [%H:%M:%S]')}                                           ┃
┃                                                                            ┃
┃ TRAINING SETTINGS.                                                         ┃
┃     OUTER FOLDS [{outer_k}]                                                        ┃
┃     INNER FOLDS [{inner_k}]                                                        ┃
┃                                                                            ┃
┃ RANDOM SEARCH SETTINGS                                                     ┃
┃     CONFIDENCE [{int((1-ALPHA)*100):02d}%]                                                       ┃
┃     PERCENTILE [{TARGET_PERCENTILE:02d}th]                                                      ┃
┃     CONFIGURATIONS [{hyperparameters['ITERATIONS']:02d}]                                                    ┃
┃     COVERAGE [{round(hyperparameters['ITERATIONS'] / prod([len(options) for options in hyperparameters["GRID"].values()]) * 100):02d}%]                                                         ┃
┃                                                                            ┃
┃ MODEL CONFIGURATION [{network_type}]{' '*(53-len(network_type))}┃""")
    for i, layer in enumerate(network):
        if i == 0: print(f"┃     {layer['type'].upper()}{' '*(71-len(layer['type']))}┃")
        else: print(f"┃     → {layer['type'].upper()}{' '*(69-len(layer['type']))}┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    


    # === OUTER FOLDS ===

    model_performances = []

    # Split data patient wise into K folds for outer CV loop
    outer_folds = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    for i_outer, (i_outer_train, i_outer_test) in enumerate(outer_folds.split(subject_list.subjects)):
        print(f"""┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ OUTER [{(i_outer+1):02d}]                                                                 ┃
┣━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┫
┃ MODEL ┃ ELAPSED ┃   LOSS   ┃   ACC   ┃  PREC  ┃   TPR   ┃   TNR   ┃   F1   ┃
┣━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┫""")

        t_outer = default_timer()

        OUTPUT = {
            "train_indices": i_outer_train.tolist(),
            "test_indices": i_outer_test.tolist(),
            "summary": {},
            "configs": {}
        }

        # Create list of custom SubjectData objects (in data/ecg_dataset.py)
        outer_train_set = [subject_list[i] for i in i_outer_train]
        outer_test_set = [subject_list[i] for i in i_outer_test]



        # === HYPERPARAMETER RANDOM SEARCH ===

        best_f1, i_best_config, best_config = 0, 0, None

        # Create hyperparameter configurations + inner fold split for inner cross validation
        configs = list(ParameterSampler(hyperparameters["GRID"], hyperparameters["ITERATIONS"], random_state=i_outer))
        for i_config, config in enumerate(configs):
            t_config = default_timer()
            f1_scores = []

            OUTPUT["configs"][i_config+1] = {
                "summary": {},
                "hyperparameters": config,
                "inner_folds": {}
            }

            # === INNER FOLDS ===

            inner_folds = KFold(n_splits=inner_k, shuffle=True, random_state=(i_outer+1)*(i_config+1))
            for i_inner, (i_inner_train, i_inner_test) in enumerate(inner_folds.split(outer_train_set)):
                print("┗━━━━━━━┷━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┷━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┛", end="\r")
                t_inner = default_timer()

                # Create inner DataLoaders for training/testing
                inner_train_loader = DataLoader(dataset=SegmentDataset([outer_train_set[i] for i in i_inner_train]), batch_size=config["BATCH_SIZE"], shuffle=True)
                inner_test_loader = DataLoader(dataset=SegmentDataset([outer_train_set[i] for i in i_inner_test]), batch_size=TEST_BATCH_SIZE)

                # Initialising Model + Optimiser
                model = DynamicCNN(network).to(device=DEVICE)
                optimiser = AdamW(params=model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

                # Training + Evaluating Model
                model = train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=config["EPOCHS"], dataloader=inner_train_loader)
                performance = evaluate_model(model=model, device=DEVICE, batch_size=TEST_BATCH_SIZE, dataloader=inner_test_loader)
                metrics = performance["metrics"]

                f1_scores.append(metrics["f1"])
                elapsed_time = default_timer()-t_inner

                OUTPUT["configs"][i_config+1]["inner_folds"][i_inner+1] = {
                    "time": elapsed_time,
                    "performance": performance
                }
                
                print(f"""┃ {(i_config+1):02d}_{(i_inner+1):02d} │ {f"{elapsed_time:.6f}"[:7]} │ {f"{performance['loss']:.7f}"[:8]} │ {f"{metrics['accuracy']*100:.6f}"[:7]} │ {f"{metrics['precision']:.5f}"[:6]} │ {f"{metrics['recall']:.6f}"[:7]} │ {f"{metrics['specificity']:.6f}"[:7]} │ {f"{metrics['f1']:.5f}"[:6]} ┃""")
                if (i_inner + 1 == inner_k) and (i_config + 1 != hyperparameters["ITERATIONS"]): print("┣━━━━━━━┿━━━━━━━━━┿━━━━━━━━━━┿━━━━━━━━━┿━━━━━━━━┿━━━━━━━━━┿━━━━━━━━━┿━━━━━━━━┫")
                elif (i_config + 1 != hyperparameters["ITERATIONS"]) or (i_inner + 1 != inner_k): print("┠───────┼─────────┼──────────┼─────────┼────────┼─────────┼─────────┼────────┨")

            mean_f1, std_f1 = mean(f1_scores), std(f1_scores)

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_config = config
                i_best_config = i_config

            OUTPUT["configs"][i_config+1]["summary"] = {
                "time": default_timer() - t_config,
                "f1": {
                    "scores": f1_scores,
                    "mean": mean_f1,
                    "std_dev": std_f1
                }
            }

        print("┣━━━━━━━┷━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┷━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┫")
        print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛", end="\r")

        t_test = default_timer()

        # Create outer DataLoaders for training/testing
        outer_train_loader = DataLoader(dataset=SegmentDataset(outer_train_set), batch_size=best_config["BATCH_SIZE"], shuffle=True)
        outer_test_loader = DataLoader(dataset=SegmentDataset(outer_test_set), batch_size=TEST_BATCH_SIZE)

        model = DynamicCNN(network).to(device=DEVICE)
        optimiser = AdamW(params=model.parameters(), lr=best_config["LR"], weight_decay=best_config["WEIGHT_DECAY"])
        model = train_model(model=model, optimiser=optimiser, device=DEVICE, epochs=best_config["EPOCHS"], dataloader=outer_train_loader)
        performance = evaluate_model(model=model, device=DEVICE, batch_size=TEST_BATCH_SIZE, dataloader=outer_test_loader)
        metrics = performance["metrics"]

        OUTPUT["summary"] = {
            "model": network,
            "time": default_timer() - t_outer,
            "best_config": best_config,
            "performance": performance
        }

        model_performances.append(performance)

        print(f"""┃ OUTER [{(i_outer+1):02d}] TOTAL TIME: {f"{(default_timer()-t_outer):.6f}"[:7]}s                                            ┃
┠───────┬─────────┬──────────┬─────────┬────────┬─────────┬─────────┬────────┨
┃ {(i_outer+1):02d}_{(i_best_config+1):02d} │ {f"{(default_timer()-t_test):.6f}"[:7]} │ {f"{performance['loss']:.7f}"[:8]} │ {f"{metrics['accuracy']*100:.6f}"[:7]} │ {f"{metrics['precision']:.5f}"[:6]} │ {f"{metrics['recall']:.6f}"[:7]} │ {f"{metrics['specificity']:.6f}"[:7]} │ {f"{metrics['f1']:.5f}"[:6]} ┃""")
        print("┗━━━━━━━┷━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┷━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━┛")



        # === SAVING MODEL ===
        if not path.exists(output_path): mkdir(output_path)
        with open(path.join(output_path, f"{i_outer+1}.json"), "w", encoding="utf8") as file: file.write(sub(r"(?<=\[)[^\[\]]+(?=\])", replace_func, dumps(OUTPUT, indent=4)))
    
    acc_scores, prec_scores, recall_scores, spec_scores, f1_scores = [], [], [], [], []

    for performance in model_performances:
        metrics = performance["metrics"]
        acc_scores.append(metrics["accuracy"])
        prec_scores.append(metrics["precision"])
        recall_scores.append(metrics["precision"])
        spec_scores.append(metrics["specificity"])
        f1_scores.append(metrics["f1"])

    if not path.exists(output_path): mkdir(output_path)
    with open(path.join(output_path, "summary.json"), "w", encoding="utf8") as file:
        json.dump({
            "time": default_timer() - t_ncv,
            "metrics": {
                "accuracy": mean(acc_scores),
                "precision": mean(prec_scores),
                "recall": mean(recall_scores),
                "specificity": mean(spec_scores),
                "f1": mean(f1_scores)
            }
        }, file, indent=4)



if __name__ == "__main__":
    ncv(inner_k=INNER_K, outer_k=OUTER_K, network_type=argv[1].upper(), hyperparameter_set=argv[2].upper(), subject_list=SUBJECT_LIST, output_path=OUTPUT_PATH, random_state=RANDOM_STATE)