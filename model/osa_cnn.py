import os
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torchview import draw_graph

class OSA_CNN(nn.Module):
    def __init__(self, conv1_config: list, conv2_config: list, conv3_config: list, linear_neurons: int, dropout: float):
        """
        PARAMETERS
        ---
        convN_config: kernel size, features, stride, and padding for Nth convolutional layer  \\
        linear_neurons: number of neurons in hidden layer
        dropout: dropout p
        """

        super(OSA_CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        k1, f1, s1, p1 = conv1_config
        k2, f2, s2, p2 = conv2_config
        k3, f3, s3, p3 = conv3_config

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=f1, kernel_size=k1, stride=s1, padding=p1)
        self.bn1 = nn.BatchNorm1d(f1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=p2)
        self.bn2 = nn.BatchNorm1d(f2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=f2, out_channels=f3, kernel_size=k3, stride=s3, padding=p3)
        self.bn3 = nn.BatchNorm1d(f3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(in_features=f3, out_features=linear_neurons)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=linear_neurons, out_features=1)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(2)
        x = self.dropout1(self.relu(self.linear1(x)))
        return self.linear2(x)

class OSA_CNN_SMALL(nn.Module):
    def __init__(self, conv1_config: list, conv2_config: list, linear_neurons: int, dropout: float):
        """
        PARAMETERS
        ---
        convN_config: kernel size, features, stride, and padding for Nth convolutional layer  \\
        linear_neurons: number of neurons in hidden layer
        dropout: dropout p
        """

        super(OSA_CNN_SMALL, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        k1, f1, s1, p1 = conv1_config
        k2, f2, s2, p2 = conv2_config

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=f1, kernel_size=k1, stride=s1, padding=p1)
        self.bn1 = nn.BatchNorm1d(f1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=p2)
        self.bn2 = nn.BatchNorm1d(f2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.linear1 = nn.Linear(in_features=f2, out_features=linear_neurons)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=linear_neurons, out_features=1)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(2)
        x = self.dropout1(self.relu(self.linear1(x)))
        return self.linear2(x)
        


# MODEL DEBUG + VISUALISATION
if __name__ == "__main__":
    model = OSA_CNN()
    summary(model, input_size=(8, 1, 6000), col_names=["input_size", "output_size", "num_params", "kernel_size"])
    if not os.path.exists("ModelVisuals") and input('Visualise the Model (Y/N): ').lower() == "y":
        os.mkdir("ModelVisuals")
        draw_graph(model, input_size=(1, 1, 6000), expand_nested=True).visual_graph.render("osa_cnn", os.path.abspath("ModelVisuals"), format="svg")
        print('Stored in ModelVisuals')