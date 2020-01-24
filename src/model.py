import torch
import torch.nn as nn


class PointCNN(nn.Module):

    def __init__(self):
        super(PointCNN, self).__init__()
        # input (batch_size, num_max_atoms, num_features)
        self.conv1 = nn.Conv1d(in_channels=29, out_channels=512, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=4)
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])

        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(x.shape[0])

        return x
