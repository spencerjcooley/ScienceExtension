import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torchview import draw_graph

class OSA_CNN(nn.Module):
    def __init__(self):
        super(OSA_CNN, self).__init__()

        # (6000, 1) | 1 Channel
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7) # 6000 -> 6000
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # 6000 -> 3000

        # (3000, 1) | 32 Channels
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3) # 3000 -> 3000
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # 3000 -> 1500

        # (1500, 1) | 64 Channels
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 1500 -> 1500
        self.bn3 = nn.BatchNorm1d(128)

        # (1500, 1) | 128 Channels
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # (128, 1) | This now corresponds to height and width
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)

        # (64, 1)
        self.fc2 = nn.Linear(64, 1)
        # -> 1

    def forward(self, x):
        # X shape: (N, 1, 6000) | Note: N is the batch size which will be used for Batch GD
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))     # (N, 32, 3000) | (Batch Size, Height, Channels)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))     # (N, 64, 1500)
        x = F.relu(self.bn3(self.conv3(x)))                 # (N, 128, 750)
        x = self.global_pool(x).squeeze(2)                  # (N, 128)
        x = self.dropout1(F.relu(self.fc1(x)))              # (N, 64)
        x = self.fc2(x)                                     # (N, 1)
        return x # No sigmoid required by using BCEWithLogitsLoss (Sigmoid built in)

model = OSA_CNN()

# MODEL DEBUG + VISUALISATION
summary(model, input_size=(8, 1, 6000), col_names=["input_size", "output_size", "num_params", "kernel_size"])
if not os.path.exists("ModelVisuals"):
    os.mkdir("ModelVisuals")
    draw_graph(model, input_size=(1, 1, 6000), expand_nested=True).visual_graph.render("osa_cnn", os.path.abspath("ModelVisuals"), format="svg")