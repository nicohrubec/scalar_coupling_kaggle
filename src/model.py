import torch
import torch.nn as nn


class PointCNN(nn.Module):

    def __init__(self):
        super(PointCNN, self).__init__()
        # input (batch_size, num_features, num_atoms)
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=29)
        self.dense1 = nn.Linear(128, 1024)
        self.dense2 = nn.Linear(1024, 4096)
        self.dense3 = nn.Linear(4096, 2048)
        self.dense4 = nn.Linear(2048, 256)
        self.dense5 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # reshape so that channels are second axis in the tensor
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dense5(x)
        x = x.view(x.shape[0])

        return x
