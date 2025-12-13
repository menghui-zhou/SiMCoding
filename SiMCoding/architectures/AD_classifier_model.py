import torch
import torch.nn as nn
import torch.nn.functional as F


class ADClassifier(nn.Module):
    def __init__(self, fd=128, n_class=2):
        super(ADClassifier, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Grayscale input
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(1024, fd)  # Adjust input size based on pooling


    def forward(self, x):
        # Convolutional layers with BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> Pool

        # Flatten for fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Flatten

        # Fully connected layers with BatchNorm and Dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Normalize feature vector
        x = F.normalize(x)
        return x


class ADClassifier_CE(nn.Module):
    def __init__(self, fd=128, n_class=2):
        super(ADClassifier_CE, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Grayscale input
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(1024, fd)  # Adjust input size based on pooling
        self.bn_fc1 = nn.BatchNorm1d(fd)  # Batch normalization for fully connected layer

        self.fc3 = nn.Linear(fd, n_class)  # Binary classification

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def extract_feature(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> Pool

        # Flatten for fully connected layer
        x = x.view(-1, 128 * 28 * 28)  # Flatten

        # Fully connected layers with BatchNorm and Dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Normalize feature vector
        x = F.normalize(x)
        return x
    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc3(x)  # Final fully connected layer
        return x


class VoxCNN(nn.Module):
    def __init__(self, fd=128, num_classes=2):
        super(VoxCNN, self).__init__()

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(13824, 128),  # Adjust input size based on the final output size of deconv layers
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.7),
            nn.Linear(128, fd),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(fd, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc(x)
        x = F.normalize(x)
        return x


# # Example usage
# model = VoxCNN(fd=128, num_classes=2)
# print(model)


class VoxCNN_CE(nn.Module):
    def __init__(self, fd=128, num_classes=2):
        super(VoxCNN_CE, self).__init__()

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(13824, 128),  # Adjust input size based on the final output size of deconv layers
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.7),
            nn.Linear(128, fd),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(fd, num_classes)

    def extract_feature(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc(x)
        x = F.normalize(x)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc2(x)
        return x
