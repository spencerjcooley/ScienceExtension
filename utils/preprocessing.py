import os
import wfdb
import numpy as np
from timeit import default_timer
from scipy.signal import butter, filtfilt

DB = os.path.abspath("ApneaECG") if os.path.exists("ApneaECG") else os.path.abspath("apnea-ecg-database-1.0.0")
if not os.path.exists("Data"): os.mkdir("Data")
OUTPUT = os.path.abspath("Data")
with open(os.path.join(DB, "list")) as file: RECORD_LIST = [record.strip() for record in file.readlines()]

def get_record(recording): return wfdb.rdrecord(os.path.join(DB, recording))
def get_labels(recording): return wfdb.rdann(os.path.join(DB, recording), 'apn')

# Precompute For BB Filter
b, a = butter(4, [0.01, 0.8], btype='band')

for recording in RECORD_LIST:
    t_0 = default_timer()

    ann, record = get_labels(recording), get_record(recording)
    signal = record.p_signal[:, 0]

    # PRE-PROCESSING
    bandpassfilter = filtfilt(b, a, signal) # Butterworth Bandpass Filter (0.5 - 40 Hz)
    output = (bandpassfilter - np.mean(bandpassfilter)) / np.std(bandpassfilter) # Z-Score Normalisation

    n_samples = 60 * record.fs

    ann_length, output_length = len(ann.symbol), len(output) // n_samples
    n_segments = min(ann_length, output_length)

    segments = np.array([output[i * n_samples : (i+1) * n_samples] for i in range(n_segments)]).astype(np.float32)
    labels = np.array([0 if ann.symbol[j] == 'N' else 1 for j in range(n_segments)]).astype(np.int8)

    np.savez_compressed(os.path.join(OUTPUT, recording), segments=segments, labels=labels)
    print(f'{recording} computed in {default_timer() - t_0}s')