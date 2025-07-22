from re import sub
from json import dumps
import numpy as np
from os import path, mkdir
from math import log, exp
from datetime import datetime
from timeit import default_timer
from warnings import filterwarnings

from torch import cuda
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, ParameterSampler
filterwarnings("ignore", category=UserWarning, module='sklearn\.model_selection\..*')

from data import SubjectList, SegmentDataset
from model import DynamicCNN, train_model, evaluate_model, FocalLoss

# Email notifications when away from computer
import smtplib
from email.mime.text import MIMEText
from personal_info import sender_email, receiver_email, password


# === TOOLS ===
def create_email_server():
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465) # 465: SSL | 587: TLS/STARTTLS
    #server.starttls()
    server.login(sender_email, password)
    return server

def send_email(server: smtplib.SMTP, subject: str, body: str):
    message = MIMEText(body, "plain")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    server.sendmail(sender_email, receiver_email, message.as_string())

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
SUBJECT_LIST = SubjectList(path.abspath("data"))

DEVICE = "cuda" if cuda.is_available() else "cpu"
OUTER_K, INNER_K = 5, 4
TEST_BATCH_SIZE = 512

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 90 # < 100

MODELS = {
    "2CONV 8|8(S)_16|5": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 8, "kernel_size": 8, "stride": 2, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 8},
        {"type": "conv1d", "in_channels": 8, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 16, "out_features": 1}
    ],
    "2CONV 16|8(S)_32|5": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 32, "out_features": 1}
    ],
    "2CONV 8|5_16|5": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 8, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 8},
        {"type": "conv1d", "in_channels": 8, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 16, "out_features": 1}
    ],
    "2CONV 16|5_32|5": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 2},
        {"type": "relu"},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "flatten"},
        {"type": "linear", "in_features": 32, "out_features": 1}
    ]
}

NCV_CONFIGS = {
    "MAIN": {
        "ITERATIONS": int(-(-log(ALPHA) // log((TARGET_PERCENTILE/100)))), # Iteration estimation via X~Bin(n,p) | Ceiling Function
        "GRID": {
            "LR": insert_logarithmic_means(start=1e-4, end=1e-3, n_means=2, is_int=False),
            "BATCH_SIZE": [64, 128],
            "EPOCHS": insert_logarithmic_means(start=50, end=100, n_means=2),
            "ALPHA": [0.65, 0.75, 0.85],
            "GAMMA": [1.5, 2],
            "THRESHOLD": [0.5, 0.65, 0.8]
        }
    },
    "DEBUG": {
        "ITERATIONS": 1,
        "GRID": {
            "LR": [1e-3],
            "BATCH_SIZE": [64],
            "EPOCHS": [10],
            "ALPHA": [0.75],
            "GAMMA": [2.0],
            "THRESHOLD": [0.6]
        }
    }
}



def cv(k: int, model_architecture: dict, config: dict, test_batch_size: int, subject_list: list, random_seed: int):
    t_config = default_timer()
    output = { "summary": {}, "inner_folds": {} }

    f1_scores = []
    loss_scores = []

    sfk = stratified_subject_split(subject_list, k, seed=random_seed)
    loss_function = FocalLoss(config["ALPHA"], config["GAMMA"], eps=1e-6)
    for i, (train_list, test_list) in enumerate(sfk, 1):
        t_fold = default_timer()

        train_loader = DataLoader(SegmentDataset(train_list), config["BATCH_SIZE"], shuffle=True)
        test_loader = DataLoader(SegmentDataset(test_list), test_batch_size, shuffle=True)

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



def ncv(outer_k: int, inner_k: int, model_name: str, model_architecture: dict, hyperparameters: dict, test_batch_size: int, subject_list: SubjectList, email_server: smtplib.SMTP, random_seed: int = 42):

    start_time = datetime.now()
    output_path = path.join(path.abspath("output"), f"data-{start_time.strftime('%Y%m%d-%H%M%S')}")
    config_iterations = hyperparameters["ITERATIONS"]

    print(f"  NCV TRIAL {start_time.strftime('%d/%m/%Y %H:%M:%S')}")

    mkdir(output_path)
    sfk = stratified_subject_split(subject_list.subjects, outer_k, random_seed)
    for i_outer, (train_list, test_list) in enumerate(sfk, 1):
        print(f"    OUTER FOLD {i_outer:02d}")
        t = default_timer()

        i_best_config, best_config = 0, None
        best_f1 = 0
        output = {
            "time": 0,
            "model": {},
            "configs": {}
        }

        configs = list(ParameterSampler(hyperparameters["GRID"], config_iterations, random_state=i_outer))
        for i_config, config in enumerate(configs, 1):            
            config_data = cv(inner_k, model_architecture, config, test_batch_size, train_list, random_seed=(i_outer*i_config))
            output["configs"][i_config] = config_data

            if config_data["summary"]["mean_f1"] > best_f1:
                best_f1 = config_data["summary"]["mean_f1"]
                i_best_config = i_config
                best_config = config
                print(f"""      CONFIG {i_config:02d}/{config_iterations} B | TIME: {f"{config_data['summary']['time']:7f}"[:8]} | MEAN F1: {f"{config_data['summary']['mean_f1']:7f}"[:8]}""")
            else: print(f"""      CONFIG {i_config:02d}/{config_iterations}   | TIME: {f"{config_data['summary']['time']:7f}"[:8]} | MEAN F1: {f"{config_data['summary']['mean_f1']:7f}"[:8]}""")

        t_model = default_timer()

        train_loader = DataLoader(SegmentDataset(train_list), best_config["BATCH_SIZE"], shuffle=True)
        test_loader = DataLoader(SegmentDataset(test_list), test_batch_size, shuffle=True)
        loss_function = FocalLoss(best_config["ALPHA"], best_config["GAMMA"], eps=1e-6)

        model = DynamicCNN(model_architecture).to(DEVICE)
        optimiser = AdamW(model.parameters(), lr=best_config["LR"])
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=best_config["LR"], steps_per_epoch=len(train_loader), epochs=best_config["EPOCHS"])

        losses = train_model(model, optimiser, scheduler, loss_function, DEVICE, best_config["EPOCHS"], train_loader)
        performance_train = evaluate_model(model, DEVICE, loss_function, train_loader, threshold=best_config["THRESHOLD"])
        performance_test = evaluate_model(model, DEVICE, loss_function, test_loader, threshold=best_config["THRESHOLD"])

        t_model = default_timer() - t_model

        output["time"] = default_timer() - t
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

        print(f"""    OUTER MODEL {i_outer:02d}   | TIME: {f"{t_model:7f}"[:8]} | F1: {f"{performance_test['metrics']['f1']:7f}"[:8]}\n""")

        subject = f"""OUTER {i_outer:02d} | {model_name} | TIME: {f"{t_model:7f}"[:8]}"""
        body = f"""TIME: {t_model}
ARCHITECTURE: {dumps(model_architecture, indent=4)}
HYPERPARAMETERS: {dumps(best_config, indent=4)}
TRAINING PERFORMANCE: {dumps(performance_train, indent=4)}
TESTING PERFORMANCE: {dumps(performance_test, indent=4)}"""
        send_email(email_server, subject, body)

        with open(path.join(output_path, f"{i_outer}.json"), "w", encoding="utf8") as file: file.write(sub(r"(?<=\[)[^\[\]]+(?=\])", replace_func, dumps(output, indent=4)))


if __name__ == "__main__":
    email_server = create_email_server()
    for model_name, model_architecture in MODELS.items():
        print(model_name)
        ncv(outer_k=OUTER_K, inner_k=INNER_K, model_name=model_name, model_architecture=model_architecture, hyperparameters=NCV_CONFIGS["MAIN"], test_batch_size=TEST_BATCH_SIZE, subject_list=SUBJECT_LIST, email_server=email_server)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")