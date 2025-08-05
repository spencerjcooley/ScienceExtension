import numpy as np
from re import sub
from json import dumps
from os import path, mkdir
from datetime import datetime
from math import log, exp, prod
import matplotlib.pyplot as plt
from timeit import default_timer
from warnings import filterwarnings

from torch import cuda, backends, empty, from_numpy
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch import set_float32_matmul_precision
from sklearn.model_selection import StratifiedKFold, ParameterSampler
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
    d = (log(end) - log(start)) / (n_means + 1)
    return [round(exp(log(start) + i * d)) for i in range(n_means + 2)] if is_int else [exp(log(start) + i * d) for i in range(n_means + 2)]

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

SUBJECT_LIST = SubjectList(path.abspath("data"))

DEVICE = "cuda" if cuda.is_available() else "cpu"
OUTER_K, INNER_K = 5, 4
TEST_BATCH_SIZE = 512

ALPHA = 0.05 # > 0
TARGET_PERCENTILE = 90 # < 100

MODELS = {
    "1": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 15, "stride": 1, "padding": 7, "bias": False},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "relu"},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "dropout", "p": 0.1},

        {"type": "conv1d", "in_channels": 16, "out_channels": 24, "kernel_size": 11, "stride": 1, "padding": 5, "bias": False},
        {"type": "batchnorm1d", "num_features": 24},
        {"type": "relu"},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "dropout", "p": 0.1},

        {"type": "conv1d", "in_channels": 24, "out_channels": 32, "kernel_size": 7, "stride": 1, "padding": 3, "bias": False},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "relu"},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "dropout", "p": 0.4},

        {"type": "flatten"},
        {"type": "linear", "in_features": 32, "out_features": 1}
    ],
    "2": [
        {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 15, "stride": 1, "padding": 7, "bias": False},
        {"type": "batchnorm1d", "num_features": 16},
        {"type": "relu"},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "dropout", "p": 0.2},

        {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 11, "stride": 1, "padding": 5, "bias": False},
        {"type": "batchnorm1d", "num_features": 32},
        {"type": "relu"},
        {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
        {"type": "dropout", "p": 0.2},

        {"type": "conv1d", "in_channels": 32, "out_channels": 64, "kernel_size": 7, "stride": 1, "padding": 3, "bias": False},
        {"type": "batchnorm1d", "num_features": 64},
        {"type": "relu"},
        {"type": "adaptiveavgpool1d", "output_size": 1},
        {"type": "dropout", "p": 0.4},

        {"type": "flatten"},
        {"type": "linear", "in_features": 64, "out_features": 1}
    ]
}

NCV_CONFIGS = {
    "MAIN": {
        "ITERATIONS": int(-(-log(ALPHA) // log((TARGET_PERCENTILE/100)))), # Iteration estimation via X~Bin(n,p) | Ceiling Function
        "GRID": {
            "LR": insert_logarithmic_means(start=1e-4, end=3e-4, n_means=2, is_int=False),
            "BATCH_SIZE": [64, 128],
            "EPOCHS": insert_logarithmic_means(start=50, end=80, n_means=1),
            "ALPHA": [0.7, 0.75, 0.8],
            "GAMMA": [1.75, 2, 2.25],
            "THRESHOLD": [0.4, 0.5, 0.6]
        }
    },
    "MAIN2": {
        "ITERATIONS": int(-(-log(ALPHA) // log((TARGET_PERCENTILE/100)))),
        "GRID": {
            "LR": [0.5e-4, 1e-4, 1.5e-4],
            "BATCH_SIZE": [64, 128],
            "EPOCHS": [50, 60],
            "ALPHA": [0.3, 0.4],
            "GAMMA": [1.3, 1.35, 1.4],
            "THRESHOLD": [0.35, 0.4, 0.45],
            "WEIGHT_DECAY": [0.5e-4, 1e-4, 1.5e-4]
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

TUNE_MODEL = [
    {"type": "conv1d", "in_channels": 1, "out_channels": 16, "kernel_size": 15, "stride": 1, "padding": 7, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 16},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.2},

    {"type": "conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 11, "stride": 1, "padding": 5, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 32},
    {"type": "maxpool1d", "kernel_size": 2, "stride": 2},
    {"type": "dropout", "p": 0.2},

    {"type": "conv1d", "in_channels": 32, "out_channels": 64, "kernel_size": 7, "stride": 1, "padding": 3, "bias": False},
    {"type": "relu"},
    {"type": "batchnorm1d", "num_features": 64},
    {"type": "adaptiveavgpool1d", "output_size": 1},
    {"type": "dropout", "p": 0.4},

    {"type": "flatten"},
    {"type": "linear", "in_features": 64, "out_features": 1}
]

TUNE_CONFIG = {
    "LR": 1e-4,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "ALPHA": 0.3,
    "GAMMA": 1.4,
    "THRESHOLD": 0.5,
    "WEIGHT_DECAY": 1e-4
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
    train_perfs, test_perfs = [], []

    t_ncv = default_timer()
    start_time = datetime.now()
    output_path = path.join(path.abspath("output"), f"data-{start_time.strftime('%Y%m%d-%H%M%S')}")
    config_iterations = hyperparameters["ITERATIONS"]

    print(f"  NCV TRIAL {start_time.strftime('%d/%m/%Y %H:%M:%S')} | ITERATIONS {config_iterations} | COVERAGE {100*config_iterations/prod([len(x) for x in hyperparameters['GRID'].values()]):3f}%")

    mkdir(output_path)
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

        print(f"""    OUTER MODEL {i_outer:02d}  | TIME: {f"{t_model:.7f}"[:8]} | F1: {f"{performance_test['metrics']['f1']:.7f}"[:8]}\n""")

        subject = f"""OUTER {i_outer:02d} | {model_name} | TIME: {f"{t_model:.7f}"[:8]}"""
        body = f"""TIME: {t_model}
ARCHITECTURE: {sub(REGEX, replace_func, dumps(model_architecture, indent=4))}

HYPERPARAMETERS: {dumps(best_config, indent=4)}

TRAINING PERFORMANCE: {dumps(performance_train, indent=4)}

TESTING PERFORMANCE: {dumps(performance_test, indent=4)}"""
        try: send_email(subject, body)
        except Exception as e: print("ERROR", e)

        with open(path.join(output_path, f"{i_outer}.json"), "w", encoding="utf8") as file: file.write(sub(REGEX, replace_func, dumps(output, indent=4)))

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

    with open(path.join(output_path, f"summary.json"), "w", encoding="utf8") as file: file.write(sub(REGEX, replace_func, dumps({
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



# === MAIN ===
if __name__ == "__main__":
    HOLDOUT_LIST = [0, 1, 22, 25, 26]
    EVAL_LIST = list(range(2, 22)) + [23, 24] + list(range(27, 35))

    HOLDOUT_SUBJECT_LIST = [SUBJECT_LIST[i] for i in HOLDOUT_LIST]
    EVAL_SUBJECT_LIST = [SUBJECT_LIST[i] for i in EVAL_LIST]

    backends.cudnn.benchmark = True
    backends.cudnn.deterministic = False
    set_float32_matmul_precision('high')

    match input('Tune OR Final (T/F): ').upper():
        case 'F':
            if not path.exists("output"): mkdir("output")

            email_string = '\n'.join([model_name for model_name in MODELS.keys()])
            try: send_email("TRAINING HAS STARTED", email_string)
            except Exception as e: print("ERROR", e)

            for model_name, model_architecture in MODELS.items():
                print(f"MODEL: {model_name}")
                ncv(outer_k=OUTER_K, inner_k=INNER_K, model_name=model_name, model_architecture=model_architecture, hyperparameters=NCV_CONFIGS["MAIN2"], test_batch_size=TEST_BATCH_SIZE, subject_list=EVAL_SUBJECT_LIST)
                print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        case 'T':
            loss_function = FocalLoss(TUNE_CONFIG["ALPHA"], TUNE_CONFIG["GAMMA"], eps=1e-6)

            def add_gaussian_noise(signal, noise_std=0.005):
                return signal + from_numpy(np.random.normal(0, noise_std, signal.shape))

            RAW_TRAINING_SET = [HOLDOUT_SUBJECT_LIST[0], HOLDOUT_SUBJECT_LIST[4]] # 1 Class A, 1 Class C
            TESTING_SET = HOLDOUT_SUBJECT_LIST[1:4] # 1 From Each Class

            TRAINING_SET = []

            for recording in RAW_TRAINING_SET:
                TRAINING_SET.append(recording)
                TRAINING_SET.append(SubjectData(add_gaussian_noise(recording.x), recording.y))
                TRAINING_SET.append(SubjectData(recording.x * empty(1).uniform_(0.9, 1.1).item(), recording.y))

            print(len(TRAINING_SET))