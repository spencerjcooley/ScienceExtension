from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch import sigmoid, no_grad


class FocalLoss(Module):
    def __init__(self, alpha, gamma, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = -alpha_t * (1 - p_t) ** self.gamma * p_t.log()

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss



loss_function = FocalLoss(alpha=0.85, gamma=2.0)

def train_model(model, optimiser, device, epochs, dataloader):
    scaler = GradScaler(device=device)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()

            with autocast(device_type=device):
                logits = model(x_batch).squeeze(1)
                loss = loss_function(logits, y_batch)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimiser)
            scaler.update()
    
    return model



def evaluate_model(model, device, batch_size, dataloader, threshold=0.5):
    total_loss, total_segments = 0.0, 0
    TP, TN, FP, FN = 0, 0, 0, 0

    model.eval()
    with no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with autocast(device_type=device):
                logits = model(x_batch).squeeze(1)
                loss = loss_function(logits, y_batch)

            total_loss += loss.item() * batch_size
            total_segments += batch_size

            predictions = (sigmoid(logits) >= threshold).float()
            TP += (predictions * y_batch).sum().item()
            TN += ((1 - predictions) * (1 - y_batch)).sum().item()
            FP += (predictions * (1 - y_batch)).sum().item()
            FN += ((1 - predictions) * y_batch).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return {
        "loss": total_loss / total_segments,
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        },
        "metrics": {
            "accuracy": (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
            "precision": precision,
            "recall": recall,
            "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
            "f1": 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        }
    }
