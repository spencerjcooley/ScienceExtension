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