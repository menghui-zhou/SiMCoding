import torch
import torch.nn as nn
import torch.nn.functional as F


class PestClassifier(nn.Module):
    def __init__(self, fd=128, n_class=2):
        super(PestClassifier, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Grayscale input
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, fd)  # Adjust input size based on pooling


    def forward(self, x):
        # Convolutional layers with BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> Pool
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> BN -> ReLU -> Pool

        # Flatten for fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Flatten

        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1(x)
        # Normalize feature vector
        x = F.normalize(x)
        return x








class PestClassifier_CE(nn.Module):
    def __init__(self, fd=128, n_class=4):
        super(PestClassifier_CE, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Grayscale input
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, fd)  # Adjust input size based on pooling
        self.bn_fc1 = nn.BatchNorm1d(fd)  # Batch normalization for fully connected layer

        self.fc2 = nn.Linear(fd, n_class)  # Binary classification



    def extract_feature(self, x):
        # Convolutional layers with BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> Pool


        # Flatten for fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Flatten

        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1(x)
        # Normalize feature vector
        x = F.normalize(x)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc2(x)  # Final fully connected layer
        return x
