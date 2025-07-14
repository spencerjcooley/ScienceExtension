import torch
import torch.nn as nn

def train_model(model, dataloader, epochs, optimiser, device):
    loss_function = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            logits = model(x_batch).squeeze(1)
            loss = loss_function(logits, y_batch)
            loss.backward()
            optimiser.step()

def evaluate_model(model, dataloader, device):
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

def evaluate_model_full(model, dataloader, device, threshold=0.5):
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
        "threshold": threshold,
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }