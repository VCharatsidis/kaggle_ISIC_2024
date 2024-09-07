import torch
import torch.nn as nn


class SkinCancerDetectionModel(nn.Module):
    def __init__(self, dropout):
        super(SkinCancerDetectionModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=16 * 96 * 96, out_features=16)
        self.bn1d = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)  # Dropout with 50% probability

        self.fc2 = nn.Linear(in_features=16, out_features=1)  # Assuming binary classification (cancer or not)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)

        # x = self.conv4(x)
        # # x = self.dropout2d(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.pool4(x)

        #print(x.shape)
        x = x.view(x.shape[0], -1)  # Flattening the tensor

        x = self.fc1(x)
        x = self.bn1d(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

