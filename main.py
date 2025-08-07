import os
import re
import sys
import math
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from timeit import default_timer

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import StratifiedKFold, ParameterSampler

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, module='sklearn\.model_selection\..*')

from data import SubjectList, SegmentDataset, SubjectData
from model import DynamicCNN, train_model, evaluate_model, FocalLoss

# Email notifications when away from computer
import smtplib
from email.mime.text import MIMEText
from personal_info import sender_email, receiver_email, password



# === TOOLS ===
def send_email(subject: str, body: str):
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465) # 465: SSL | 587: TLS/STARTTLS
    server.login(sender_email, password)
    message = MIMEText(body, "plain")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

def insert_logarithmic_means(start: float, end: float, n_means: int, is_int: bool = True):
    d = (math.log(end) - math.log(start)) / (n_means + 1)
    return [round(math.exp(math.log(start) + i * d)) for i in range(n_means + 2)] if is_int else [math.exp(math.log(start) + i * d) for i in range(n_means + 2)]

def insert_arithmetic_means(start: int, end: int, n_means: int, is_int: bool = True):
    d = (end - start)/(n_means + 1)
    return [round(start + i * d) for i in range(n_means + 2)] if is_int else [start + i * d for i in range(n_means + 2)]

def stratified_subject_split(subject_list: list, n_splits: int = 5, seed: int = 42):
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
REGEX = r"(?<=\[)[^\[\]]+(?=\])"

SUBJECT_LIST = SubjectList(os.path.abspath("data"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTER_K, INNER_K = 5, 4
TEST_BATCH_SIZE = 512

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 90 # < 100

TUNE_MODEL = [
    {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 13, "stride": 1, "padding": 6, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 16},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.4},

    {"type": "conv1d", "in_channels": 16, "out_channels": 24, "kernel_size": 9, "stride": 1, "padding": 4, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 24},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.4},

    {"type": "conv1d", "in_channels": 24, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 32},
    {"type": "adaptiveavgpool1d", "output_size": 1},
    {"type": "dropout", "p": 0.5},

    {"type": "flatten"},
    {"type": "linear", "in_features": 32, "out_features": 1}
]

TUNE_CONFIG = {
    "LR": 5e-4,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "ALPHA": 0.3,
    "GAMMA": 1.4,
    "THRESHOLD": 0.4,
    "WEIGHT_DECAY": 1e-4
}

FINAL_MODEL = [
    {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 13, "stride": 1, "padding": 6, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 16},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.4},

    {"type": "conv1d", "in_channels": 16, "out_channels": 24, "kernel_size": 9, "stride": 1, "padding": 4, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 24},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.4},

    {"type": "conv1d", "in_channels": 24, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 32},
    {"type": "adaptiveavgpool1d", "output_size": 1},
    {"type": "dropout", "p": 0.5},

    {"type": "flatten"},
    {"type": "linear", "in_features": 32, "out_features": 1}
]

FINAL_CONFIG = {
    "ITERATIONS": int(-(-math.log(ALPHA) // math.log((TARGET_PERCENTILE/100)))), # Iteration estimation via X~Bin(n,p) | Ceiling Function
    "GRID": {
        "LR": [1e-4, 2.5e-4, 5e-4],
        "BATCH_SIZE": [64, 128],
        "EPOCHS": [50, 60],
        "ALPHA": [0.3, 0.35],
        "GAMMA": [1.3, 1.4],
        "THRESHOLD": [0.35, 0.4, 0.45],
        "WEIGHT_DECAY": [1e-4, 1.5e-4]
    }
}



# === TRAINING FUNCTIONS ===
def cv(k: int, model_architecture: dict, config: dict, test_batch_size: int, subject_list: list, random_seed: int):
    t_config = default_timer()
    output = { "summary": {}, "inner_folds": {} }

    f1_scores = []
    loss_scores = []

    sfk = stratified_subject_split(subject_list, k, seed=random_seed)
    loss_function = FocalLoss(config["ALPHA"], config["GAMMA"], eps=1e-6)
    for i, (train_list, test_list) in enumerate(sfk, 1):
        t_fold = default_timer()

        train_loader = DataLoader(SegmentDataset(train_list), config["BATCH_SIZE"], shuffle=True, pin_memory=True)
        test_loader = DataLoader(SegmentDataset(test_list), test_batch_size, shuffle=False, pin_memory=True)

        model = DynamicCNN(model_architecture).to(DEVICE)
        optimiser = AdamW(model.parameters(), lr=config["LR"])
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=config["LR"], steps_per_epoch=len(train_loader), epochs=config["EPOCHS"])

        losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, config["EPOCHS"], train_loader)
        performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=config["THRESHOLD"])
        performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=config["THRESHOLD"])

        output["inner_folds"][i] = {
            "time": default_timer() - t_fold,
            "losses": losses,
            "train_perf": performance_train,
            "test_perf": performance_test
        }

        f1_scores.append(performance_test["metrics"]["f1"])
        loss_scores.append(performance_test["loss"])

    output["summary"] = {
        "time": default_timer() - t_config,
        "mean_loss": float(np.mean(loss_scores)),
        "mean_f1": float(np.mean(f1_scores))
    }
    return output



def ncv(outer_k: int, inner_k: int, model_name: str, model_architecture: dict, hyperparameters: dict, test_batch_size: int, subject_list: SubjectList, random_seed: int = 42):
    train_perfs, test_perfs, best_configs = [], [], []

    t_ncv = default_timer()
    start_time = datetime.now()
    output_path = os.path.join(os.path.abspath("output"), f"data-{start_time.strftime('%Y%m%d-%H%M%S')}")
    config_iterations = hyperparameters["ITERATIONS"]

    print(f"  NCV TRIAL {start_time.strftime('%d/%m/%Y %H:%M:%S')} | ITERATIONS {config_iterations} | COVERAGE {100*config_iterations/math.prod([len(x) for x in hyperparameters['GRID'].values()]):3f}%")

    os.mkdir(output_path)
    sfk = stratified_subject_split(subject_list, outer_k, random_seed)
    for i_outer, (train_list, test_list) in enumerate(sfk, 1):
        print(f"    OUTER FOLD {i_outer:02d}")
        t_outer = default_timer()

        i_best_config, best_config = 0, None
        best_f1 = 0
        output = {
            "total_time": 0,
            "model": {},
            "configs": {}
        }

        # === INNER FOLD RANDOM SEARCH ===
        configs = list(ParameterSampler(hyperparameters["GRID"], config_iterations, random_state=np.random.RandomState(random_seed+i_outer)))
        for i_config, config in enumerate(configs, 1):            
            config_data = cv(inner_k, model_architecture, config, test_batch_size, train_list, random_seed=random_seed)
            output["configs"][i_config] = config_data

            if config_data["summary"]["mean_f1"] > best_f1:
                best_f1 = config_data["summary"]["mean_f1"]
                i_best_config = i_config
                best_config = config
                print(f"""      CONFIG {i_config:02d}/{config_iterations} B | TIME: {f"{config_data['summary']['time']:.7f}"[:8]} | MEAN F1: {f"{config_data['summary']['mean_f1']:.7f}"[:8]}""")
            else: print(f"""      CONFIG {i_config:02d}/{config_iterations}   | TIME: {f"{config_data['summary']['time']:.7f}"[:8]} | MEAN F1: {f"{config_data['summary']['mean_f1']:.7f}"[:8]}""")

        t_model = default_timer()

        train_loader = DataLoader(SegmentDataset(train_list), best_config["BATCH_SIZE"], shuffle=True, pin_memory=True)
        test_loader = DataLoader(SegmentDataset(test_list), test_batch_size, shuffle=False, pin_memory=True)
        loss_function = FocalLoss(best_config["ALPHA"], best_config["GAMMA"], eps=1e-6)

        model = DynamicCNN(model_architecture).to(DEVICE)
        optimiser = AdamW(model.parameters(), lr=best_config["LR"])
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=best_config["LR"], steps_per_epoch=len(train_loader), epochs=best_config["EPOCHS"])

        losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, best_config["EPOCHS"], train_loader)
        performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=best_config["THRESHOLD"])
        performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=best_config["THRESHOLD"])
        train_perfs.append(performance_train)
        test_perfs.append(performance_test)
        best_configs.append(best_config)

        t_model = default_timer() - t_model

        output["total_time"] = default_timer() - t_outer
        output["model"] = {
            "time": t_model,
            "architecture": model_architecture,
            "config": {
                "id": i_best_config,
                "hyperparameters": best_config
            },
            "losses": losses,
            "train_perf": performance_train,
            "test_perf": performance_test
        }

        # === LOGGING + SAVING DATA ===
        print(f"""    OUTER MODEL {i_outer:02d}  | TIME: {f"{t_model:.7f}"[:8]} | F1: {f"{performance_test['metrics']['f1']:.7f}"[:8]}\n""")
        subject = f"""OUTER {i_outer:02d} | {model_name} | TIME: {f"{t_model:.7f}"[:8]}"""
        body = f"""TIME: {t_model}
ARCHITECTURE: {re.sub(REGEX, replace_func, json.dumps(model_architecture, indent=4))}

HYPERPARAMETERS: {json.dumps(best_config, indent=4)}

TRAINING PERFORMANCE: {json.dumps(performance_train, indent=4)}

TESTING PERFORMANCE: {json.dumps(performance_test, indent=4)}"""
        try: send_email(subject, body)
        except Exception as e: print("ERROR", e)
        with open(os.path.join(output_path, f"{i_outer}.json"), "w", encoding="utf8") as file: file.write(re.sub(REGEX, replace_func, json.dumps(output, indent=4)))

    # === MEAN METRICS ===
    test_accuracy = np.mean([perf["metrics"]["accuracy"] for perf in test_perfs])
    test_precision = np.mean([perf["metrics"]["precision"] for perf in test_perfs])
    test_recall = np.mean([perf["metrics"]["recall"] for perf in test_perfs])
    test_specificity = np.mean([perf["metrics"]["specificity"] for perf in test_perfs])
    test_f1 = np.mean([perf["metrics"]["f1"] for perf in test_perfs])

    train_accuracy = np.mean([perf["metrics"]["accuracy"] for perf in train_perfs])
    train_precision = np.mean([perf["metrics"]["precision"] for perf in train_perfs])
    train_recall = np.mean([perf["metrics"]["recall"] for perf in train_perfs])
    train_specificity = np.mean([perf["metrics"]["specificity"] for perf in train_perfs])
    train_f1 = np.mean([perf["metrics"]["f1"] for perf in train_perfs])

    t_ncv = default_timer() - t_ncv

    subject = f"""NCV LOOP {model_name} AVERAGE RESULTS"""
    body = f"""TIME: {t_ncv}

TRAIN ACCURACY: {train_accuracy}
TRAIN PRECISION: {train_precision}
TRAIN RECALL: {train_recall}
TRAIN SPECIFICITY: {train_specificity}
TRAIN F1: {train_f1}

TEST ACCURACY: {test_accuracy}
TEST PRECISION: {test_precision}
TEST RECALL: {test_recall}
TEST SPECIFICITY: {test_specificity}
TEST F1: {test_f1}"""
    try: send_email(subject, body)
    except Exception as e: print("ERROR", e)

    print(f"\n{body}\n")

    with open(os.path.join(output_path, f"ncv_summary.json"), "w", encoding="utf8") as file: file.write(re.sub(REGEX, replace_func, json.dumps({
        "time": t_ncv,
        "mean_train_perf": {
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
            "specificity": train_specificity,
            "f1": train_f1
        },
        "mean_test_perf": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "specificity": test_specificity,
            "f1": test_f1
        }
    }, indent=4)))
        
    # === OPTIMAL CONFIG ===
    BEST_LR = Counter([config["LR"] for config in best_configs])
    BEST_BATCH_SIZE = Counter([config["BATCH_SIZE"] for config in best_configs])
    BEST_EPOCHS = Counter([config["EPOCHS"] for config in best_configs])
    BEST_ALPHA = Counter([config["ALPHA"] for config in best_configs])
    BEST_GAMMA = Counter([config["GAMMA"] for config in best_configs])
    BEST_THRESHOLD = Counter([config["THRESHOLD"] for config in best_configs])
    BEST_WEIGHT_DECAY = Counter([config["WEIGHT_DECAY"] for config in best_configs])

    return output_path, {
        "LR": max([key for key, value in BEST_LR.items() if value == BEST_LR.most_common()[0][1]]),
        "BATCH_SIZE": max([key for key, value in BEST_BATCH_SIZE.items() if value == BEST_BATCH_SIZE.most_common()[0][1]]),
        "EPOCHS": max([key for key, value in BEST_EPOCHS.items() if value == BEST_EPOCHS.most_common()[0][1]]),
        "ALPHA": max([key for key, value in BEST_ALPHA.items() if value == BEST_ALPHA.most_common()[0][1]]),
        "GAMMA": max([key for key, value in BEST_GAMMA.items() if value == BEST_GAMMA.most_common()[0][1]]),
        "THRESHOLD": max([key for key, value in BEST_THRESHOLD.items() if value == BEST_THRESHOLD.most_common()[0][1]]),
        "WEIGHT_DECAY": max([key for key, value in BEST_WEIGHT_DECAY.items() if value == BEST_WEIGHT_DECAY.most_common()[0][1]])
    }



# === MAIN ===
if __name__ == "__main__":
    HOLDOUT_LIST = [0, 1, 22, 25, 26]
    EVAL_LIST = list(range(2, 22)) + [23, 24] + list(range(27, 35))

    HOLDOUT_SUBJECT_LIST = [SUBJECT_LIST[i] for i in HOLDOUT_LIST]
    NCV_SUBJECT_LIST = [SUBJECT_LIST[i] for i in EVAL_LIST]
    TEST_SUBJECT_LIST = SUBJECT_LIST[35:]

    torch.set_float32_matmul_precision('high')

    match sys.argv[1].upper():
        case 'FINAL':
            t_TOTAL = default_timer()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            if not os.path.exists("output"): os.mkdir("output")
            output_path, output = ncv(outer_k=OUTER_K, inner_k=INNER_K, model_name="FINAL", model_architecture=FINAL_MODEL, hyperparameters=FINAL_CONFIG, test_batch_size=TEST_BATCH_SIZE, subject_list=NCV_SUBJECT_LIST)
            
            TRAINING_SUBJECT_LIST = HOLDOUT_SUBJECT_LIST + NCV_SUBJECT_LIST

            train_loader = DataLoader(SegmentDataset(TRAINING_SUBJECT_LIST), output["BATCH_SIZE"], shuffle=True, pin_memory=True)
            test_loader = DataLoader(SegmentDataset(TEST_SUBJECT_LIST), TEST_BATCH_SIZE, shuffle=False, pin_memory=True)
            loss_function = FocalLoss(output["ALPHA"], output["GAMMA"], eps=1e-6)

            model = DynamicCNN(FINAL_MODEL).to(DEVICE)
            optimiser = AdamW(model.parameters(), lr=output["LR"])
            scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=output["LR"], steps_per_epoch=len(train_loader), epochs=output["EPOCHS"] * ((1-1/OUTER_K)*(1-1/INNER_K)*30))

            losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, output["EPOCHS"] * ((1-1/OUTER_K)*(1-1/INNER_K)*30), train_loader)
            performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=output["THRESHOLD"])
            performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=output["THRESHOLD"])
            print(f"TRAIN: {json.dumps(performance_train['metrics'], indent=4)}")
            print(f"TEST : {json.dumps(performance_test['metrics'], indent=4)}")
            
            with open(os.path.join(output_path, f"summary.json"), "w", encoding="utf8") as file: file.write(re.sub(REGEX, replace_func, json.dumps({
                "hyperparameters": output,
                "losses": losses,
                "train_perf": performance_train,
                "test_perf": performance_test
            }, indent=4)))

            try: send_email(f"FINAL EVALUATION COMPLETED", f"TIME: {default_timer() - t_TOTAL}\n\n{json.dumps({'TRAIN': performance_train, 'TEST:': performance_test}, indent=4)}")
            except Exception as e: print("ERROR", e)
        
        case 'TEST_EVAL':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            TRAINING_SUBJECT_LIST = HOLDOUT_SUBJECT_LIST + NCV_SUBJECT_LIST

            # === OBTAIN BEST CONFIGS (OVERWRITTEN SUMMARY FILE) ===
            output_files_path = os.path.join(os.path.abspath("output"), os.listdir(os.path.abspath("output"))[-1])
            best_configs = []
            for filename in os.listdir(output_files_path):
                with open(os.path.join(output_files_path, filename), 'r') as file: best_configs.append(json.load(file)["model"]["config"]["hyperparameters"])

            BEST_LR = Counter([config["LR"] for config in best_configs])
            BEST_BATCH_SIZE = Counter([config["BATCH_SIZE"] for config in best_configs])
            BEST_EPOCHS = Counter([config["EPOCHS"] for config in best_configs])
            BEST_ALPHA = Counter([config["ALPHA"] for config in best_configs])
            BEST_GAMMA = Counter([config["GAMMA"] for config in best_configs])
            BEST_THRESHOLD = Counter([config["THRESHOLD"] for config in best_configs])
            BEST_WEIGHT_DECAY = Counter([config["WEIGHT_DECAY"] for config in best_configs])

            output = {
                "LR": max([key for key, value in BEST_LR.items() if value == BEST_LR.most_common()[0][1]]),
                "BATCH_SIZE": max([key for key, value in BEST_BATCH_SIZE.items() if value == BEST_BATCH_SIZE.most_common()[0][1]]),
                "EPOCHS": round(max([key for key, value in BEST_EPOCHS.items() if value == BEST_EPOCHS.most_common()[0][1]]) * ((1-1/OUTER_K)*(1-1/INNER_K)*30/35)), # Adjust to larger test set size 
                "ALPHA": max([key for key, value in BEST_ALPHA.items() if value == BEST_ALPHA.most_common()[0][1]]),
                "GAMMA": max([key for key, value in BEST_GAMMA.items() if value == BEST_GAMMA.most_common()[0][1]]),
                "THRESHOLD": max([key for key, value in BEST_THRESHOLD.items() if value == BEST_THRESHOLD.most_common()[0][1]]),
                "WEIGHT_DECAY": max([key for key, value in BEST_WEIGHT_DECAY.items() if value == BEST_WEIGHT_DECAY.most_common()[0][1]])
            }

            train_loader = DataLoader(SegmentDataset(TRAINING_SUBJECT_LIST), output["BATCH_SIZE"], shuffle=True, pin_memory=True)
            test_loader = DataLoader(SegmentDataset(TEST_SUBJECT_LIST), TEST_BATCH_SIZE, shuffle=False, pin_memory=True)
            loss_function = FocalLoss(output["ALPHA"], output["GAMMA"], eps=1e-6)

            model = DynamicCNN(FINAL_MODEL).to(DEVICE)
            optimiser = AdamW(model.parameters(), lr=output["LR"])
            scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=output["LR"], steps_per_epoch=len(train_loader), epochs=output["EPOCHS"])

            losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, output["EPOCHS"], train_loader)
            performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=output["THRESHOLD"])
            performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=output["THRESHOLD"])
            print(f"TRAIN: {json.dumps(performance_train['metrics'], indent=4)}")
            print(f"TEST : {json.dumps(performance_test['metrics'], indent=4)}\n")
            
            # === RECORDING WISE TEST ===
            recording_wise_path = os.path.join(output_files_path, "recording_wise")
            os.mkdir(recording_wise_path)

            for i, recording in enumerate(TEST_SUBJECT_LIST, 1):
                test_loader = DataLoader(SegmentDataset([recording]), TEST_BATCH_SIZE, shuffle=False, pin_memory=True)
                recording_performance = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=output["THRESHOLD"])
                with open(os.path.join(recording_wise_path, f"x{i:02d}.json"), "w", encoding="utf8") as file: file.write(re.sub(REGEX, replace_func, json.dumps(recording_performance, indent=4)))
            torch.save(model.state_dict(), os.path.join(recording_wise_path, "model.pt"))
            

    
        case 'NCV':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            if not os.path.exists("output"): os.mkdir("output")

            ncv(outer_k=OUTER_K, inner_k=INNER_K, model_name="NCV", model_architecture=FINAL_MODEL, hyperparameters=FINAL_CONFIG, test_batch_size=TEST_BATCH_SIZE, subject_list=NCV_SUBJECT_LIST)
            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        case 'TUNE':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            loss_function = FocalLoss(TUNE_CONFIG["ALPHA"], TUNE_CONFIG["GAMMA"], eps=1e-6)

            def add_gaussian_noise(signal, noise_std=0.005):
                return signal + torch.from_numpy(np.random.normal(0, noise_std, signal.shape))

            RAW_TRAINING_SET = [HOLDOUT_SUBJECT_LIST[0], HOLDOUT_SUBJECT_LIST[4]] # 1 Class A, 1 Class C
            TESTING_SET = HOLDOUT_SUBJECT_LIST[1:4] # 1 From Each Class

            TRAINING_SUBJECT_LIST = []

            for recording in RAW_TRAINING_SET:
                TRAINING_SUBJECT_LIST.append(recording)
                TRAINING_SUBJECT_LIST.append(SubjectData(add_gaussian_noise(recording.x), recording.y))
                TRAINING_SUBJECT_LIST.append(SubjectData(recording.x * torch.empty(1).uniform_(0.9, 1.1).item(), recording.y))

            train_loader = DataLoader(SegmentDataset(TRAINING_SUBJECT_LIST), TUNE_CONFIG["BATCH_SIZE"], shuffle=True, pin_memory=True)
            test_loader = DataLoader(SegmentDataset(TESTING_SET), TEST_BATCH_SIZE, shuffle=False, pin_memory=True)
            loss_function = FocalLoss(TUNE_CONFIG["ALPHA"], TUNE_CONFIG["GAMMA"], eps=1e-6)

            model = DynamicCNN(TUNE_MODEL).to(DEVICE)
            optimiser = AdamW(model.parameters(), lr=TUNE_CONFIG["LR"])
            scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=TUNE_CONFIG["LR"], steps_per_epoch=len(train_loader), epochs=TUNE_CONFIG["EPOCHS"])

            losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, TUNE_CONFIG["EPOCHS"], train_loader)
            performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=TUNE_CONFIG["THRESHOLD"])
            performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=TUNE_CONFIG["THRESHOLD"])
            print(f"TRAIN: {json.dumps(performance_train['metrics'], indent=4)}")
            print(f"TEST : {json.dumps(performance_test['metrics'], indent=4)}")

            plt.plot(range(1, len(losses)+1), losses)
            plt.xlabel("Epochs")
            plt.ylabel("Focal Loss")
            plt.title(f"Epoch vs Loss")
            plt.show()