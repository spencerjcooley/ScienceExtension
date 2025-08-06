import os
import wfdb
import numpy as np
from timeit import default_timer
from scipy.signal import butter, filtfilt

if __name__ == "__main__":
    # === LOAD RAW TEST LABELS FROM event_answers ===
    with open(os.path.join(os.path.abspath("apnea-ecg-database-1.0.0"), "event_answers"), "r") as file:
        sequence = []
        labels = []
        for line in file:
            if line[0] != "x": sequence.append(line.strip()[3:])
            else:
                labels.append(sequence)
                sequence = []
        labels.append(sequence)
    del labels[0] # Remove initial blank
    test_labels = [''.join(label) for label in labels]

    # === VERIFY PATHS ===
    if os.path.exists("apnea-ecg-database-1.0.0"): DB = os.path.abspath("apnea-ecg-database-1.0.0")
    elif os.path.exists("ApneaECG"): DB = os.path.abspath("ApneaECG")
    else: raise RuntimeError("ECG database not found.")
    if not os.path.exists("data"): os.mkdir("data")
    OUTPUT = os.path.abspath("data")

    # === TOOLS ===
    def get_record(recording): return wfdb.rdrecord(os.path.join(DB, recording))
    def get_labels(recording): return wfdb.rdann(os.path.join(DB, recording), 'apn')

    # === PROCESS LOOP ===
    with open(os.path.join(DB, "list")) as file: RECORD_LIST = [record.strip() for record in file.readlines()]
    b, a = butter(4, [0.01, 0.8], btype='band')
    test_index = 0
    for recording in RECORD_LIST:
        t_0 = default_timer()
        record = get_record(recording)
        signal = record.p_signal[:, 0]
        n_samples = int(60 * record.fs)

        filtered = filtfilt(b, a, signal)
        normalised = (filtered - np.mean(filtered)) / np.std(filtered)

        segments, labels = [], []

        if recording[0] != "x":
            ann = get_labels(recording)
            for idx, label_symbol in enumerate(ann.symbol):
                start = ann.sample[idx]
                end = start + n_samples
                if end <= len(normalised):
                    segment = normalised[start:end]
                    label = 0 if label_symbol == 'N' else 1
                    segments.append(segment)
                    labels.append(label)
        else:
            true_sequence = test_labels[test_index]
            for idx, label_symbol in enumerate(true_sequence):
                start = idx * n_samples
                end = start + n_samples
                if end <= len(normalised):
                    segment = normalised[start:end]
                    label = 0 if label_symbol == 'N' else 1
                    segments.append(segment)
                    labels.append(label)
            test_index += 1

        segments, labels = np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int8)
        np.savez_compressed(os.path.join(OUTPUT, recording), segments=segments, labels=labels)
        print(f'{recording} computed in {default_timer() - t_0:.2f}s')
