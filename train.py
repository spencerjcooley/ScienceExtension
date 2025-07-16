import torch
import torch.nn as nn
from copy import deepcopy

def train_model(model, optimiser, device, epochs, patience, dataloader, val_dataloader=None):
    loss_function = nn.BCEWithLogitsLoss()
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improvement = 0

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            logits = model(x_batch).squeeze(1)
            loss = loss_function(logits, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimiser.step()

        if val_dataloader is not None:
            val_loss = evaluate_model(model=model, device=device, dataloader=val_dataloader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improvement = 0
            else: epochs_no_improvement += 1

            if epochs_no_improvement >= patience: return epoch + 1

        if val_dataloader is not None and best_model_state is not None: model.load_state_dict(best_model_state)

    return epochs

def evaluate_model(model, device, dataloader):
    loss_function = nn.BCEWithLogitsLoss()
    total_loss, total_segments = 0.0, 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch).squeeze(1)
            loss = loss_function(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            total_segments += x_batch.size(0)
    
    return total_loss / total_segments

def evaluate_model_full(model, device, dataloader, threshold=0.5):
    loss_function = nn.BCEWithLogitsLoss()
    total_loss, total_segments = 0.0, 0
    TP, TN, FP, FN = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch).squeeze(1)
            loss = loss_function(logits, y_batch)

            # Calculate Loss
            total_loss += loss.item() * x_batch.size(0)
            total_segments += x_batch.size(0)

            # Get Confusion Matrix
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            TP += ((predictions == 1) & (y_batch == 1)).sum().item()
            TN += ((predictions == 0) & (y_batch == 0)).sum().item()
            FP += ((predictions == 1) & (y_batch == 0)).sum().item()
            FN += ((predictions == 0) & (y_batch == 1)).sum().item()
    
    accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP + TN + FP + FN) > 0 else 0
    precision = (TP)/(TP+FP) if (TP + FP) > 0 else 0
    recall = (TP)/(TP+FN) if (TP + FN) > 0 else 0
    specificity = (TN)/(TN+FP) if (TN + FP) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if precision + recall > 0 else 0

    return {
        "loss": total_loss / total_segments,
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1
        }
    }