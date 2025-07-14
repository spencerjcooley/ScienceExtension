import wfdb
import numpy as np
import os

DB = os.path.abspath("ApneaECG")
OUTPUT = os.path.abspath("Data")
with open(os.path.join(DB, "list")) as file: RECORD_LIST = [record.strip() for record in file.readlines()]

def get_record(recording): return wfdb.rdrecord(os.path.join(DB, recording))
def get_labels(recording): return wfdb.rdann(os.path.join(DB, recording), 'apn')

for recording in RECORD_LIST:
    ann, record = get_labels(recording), get_record(recording)
    signal = record.p_signal[:, 0]
    
    n_samples = 60 * record.fs

    ann_length, signal_length = len(ann.symbol), len(signal) // n_samples
    n_segments = min(ann_length, signal_length)

    segments = np.array([signal[i * n_samples : (i+1) * n_samples] for i in range(n_segments)])
    labels = np.array([0 if ann.symbol[j] == 'N' else 1 for j in range(n_segments)])

    np.savez_compressed(os.path.join(OUTPUT, recording), segments=segments, labels=labels)
    print(f'{recording} computed')