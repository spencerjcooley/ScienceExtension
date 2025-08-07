import os
import re
import json
import numpy as np

REGEX = r"(?<=\[)[^\[\]]+(?=\])"
replace_func = lambda match: " ".join(match.group().split())

NCV_CONFUSION_MATRICES = [
    { "TP": 567, "TN": 1583, "FP": 331, "FN": 507 },
    { "TP": 710, "TN": 1270, "FP": 309, "FN": 685 },
    { "TP": 563, "TN": 1423, "FP": 236, "FN": 657 },
    { "TP": 722, "TN": 1571, "FP": 328, "FN": 165 },
    { "TP": 892, "TN": 987, "FP": 987, "FN": 79 }
]

# === Determine Majority Class ===
total_segments, total_apnea = 0, 0
for filename in os.listdir(os.path.abspath("data")):
    if not filename.endswith(".npz") or not filename.startswith("x"): continue
    data = np.load(os.path.join(os.path.abspath("data"), filename))
    labels = data["labels"]
    total_segments += len(labels)
    total_apnea += labels.sum()
majority_class = "Apnea" if total_apnea / total_segments >= 0.5 else "Normal"


# === Obtain Evaluation Data ===
data_path = os.path.join(os.path.abspath("output"), os.listdir(os.path.abspath("output"))[-1], "recording_wise")

model_accs, naive_accs = [], []

for filename in os.listdir(data_path):
    if not filename.endswith(".json") or not filename.startswith("x"): continue

    with open(os.path.join(data_path, filename), 'r') as file:
        data = json.load(file)
        confusion_matrix = data["confusion_matrix"]
        TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
        total = sum(confusion_matrix.values())

        model_accs.append((TP + TN) / total)
        naive_accs.append((FP + TN) / total)


model_accs, naive_accs = np.array(model_accs), np.array(naive_accs)

# === Paired Permutation Test ===
def permutation_test(model_metrics, naive_metrics, num_permutations=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(model_metrics - naive_metrics)

    diffs = model_metrics - naive_metrics
    permuted_diffs = np.empty(num_permutations)
    for i in range(num_permutations):
        signs = rng.choice([1, -1], size=len(diffs))
        permuted_diffs[i] = np.mean(diffs * signs)

    p_value = np.mean(permuted_diffs >= observed_diff) # One-tailed
    return observed_diff, p_value


num_permutations = 1_000_000
observed_diff, p_value = permutation_test(model_accs, naive_accs, num_permutations)

print("=== Permutation Test Results ===")
print(f"Observed Mean Difference: {observed_diff:.4f}")
print(f"One-tailed p-value: {p_value:.6f}\n")

# with open(os.path.join(data_path, "summary.json"), "w", encoding="utf8") as file: file.write(re.sub(REGEX, replace_func, json.dumps({
#     "permutations": num_permutations,
#     "observed_mean_diff": observed_diff,
#     "p": p_value
# }, indent=4)))
    
model_accs, naive_accs = [], []
for confusion_matrix in NCV_CONFUSION_MATRICES:
    total, TP, TN, FP, FN = sum(confusion_matrix.values()), confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
    model_accs.append((TP + TN) / total)
    naive_accs.append((FP + TN) / total)
model_accs, naive_accs = np.array(model_accs), np.array(naive_accs)

observed_diff, p_value = permutation_test(model_accs, naive_accs, num_permutations)
print("=== NCV Permutation Test Results ===")
print(f"Observed Mean Difference: {observed_diff:.4f}")
print(f"One-tailed p-value: {p_value:.6f}")