import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicBlock(nn.Module):
    # Expansion factor used to adjust the number of output channels. For BasicBlock, it's always 1.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer with kernel size 3x3, stride as per argument, padding=1 to maintain spatial size, and no bias.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)  # watch out, no bias
        # Batch normalization for the output of the first convolutional layer.
        self.bn1 = nn.BatchNorm2d(planes)
        # Second convolutional layer, similar to the first but always with stride=1 to maintain size.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection to allow identity mapping.
        self.shortcut = nn.Sequential()
        # If stride is not 1 or in_planes does not equal planes * expansion, adjust dimensions via 1x1 convolutions.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Forward pass through the first conv layer, then batch normalization, then ReLU activation.
        out = F.relu(self.bn1(self.conv1(x)))
        # Forward pass through the second conv layer and then batch normalization.
        out = self.bn2(self.conv2(out))
        # Addition of shortcut connection output and output from the second BN layer before applying ReLU.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # Expansion factor for bottleneck blocks, typically set to 4.
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 convolution that reduces the dimensionality.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 convolution, the core of the bottleneck, with stride and padding to maintain or reduce spatial dimensions.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 convolution that increases the dimensionality back to planes * expansion.
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection similar to BasicBlock but adjusted for the bottleneck's output size.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Sequentially apply conv1->bn1->ReLU, conv2->bn2->ReLU, and conv3->bn3 to the input.
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Add the shortcut connection output before applying the final ReLU.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """
    ResNet-50 backbone with a two-FC projector to produce features, plus an optional classifier.

    - Backbone: torchvision resnet50 (ImageNet-pretrained if pretrained=True)
    - Projector (mimics your reshape block):
        Linear(2048 -> 512, bias=False) + BN + ReLU + Linear(512 -> feature_dim)
        Outputs L2-normalized features.
    - Classifier (optional): Linear(feature_dim -> num_classes)
    """

    def __init__(self, feature_dim=512, num_classes=None, pretrained=True, normalize_feats=True):
        super().__init__()
        self.normalize_feats = normalize_feats

        # Load pretrained ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)

        # Keep everything up to the penultimate layer (avgpool)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # We’ll use adaptive avgpool to (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 is ResNet-50 final channel size
        in_dim = 2048

        # Two-FC projector (mimic your example: Linear -> BN -> ReLU -> Linear)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, feature_dim, bias=True)
        )

    def forward(self, x, return_features=False, return_logits=False):
        """
        Args:
            return_features (bool): return the projected feature (L2-normalized if normalize_feats=True)
            return_logits (bool): return classifier logits (requires num_classes not None)
        Returns:
            If both flags False: returns features (default).
            If both True: returns (features, logits).
            If only one True: returns that item.
        """
        # Backbone
        x = self.stem(x)
        x = self.avgpool(x)  # (N, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (N, 2048)

        # Project to feature_dim
        feats = self.projector(x)  # (N, feature_dim)
        if self.normalize_feats:
            feats = F.normalize(feats, dim=1)

        return feats


class ResNet50_CE(nn.Module):
    """
    ResNet-50 backbone with a two-FC projector to produce features, plus an optional classifier.

    - Backbone: torchvision resnet50 (ImageNet-pretrained if pretrained=True)
    - Projector (mimics your reshape block):
        Linear(2048 -> 512, bias=False) + BN + ReLU + Linear(512 -> feature_dim)
        Outputs L2-normalized features.
    - Classifier (optional): Linear(feature_dim -> num_classes)
    """

    def __init__(self, feature_dim=512, num_classes=None, pretrained=True, normalize_feats=True):
        super().__init__()
        self.normalize_feats = normalize_feats

        # Load pretrained ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)

        # Keep everything up to the penultimate layer (avgpool)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # We’ll use adaptive avgpool to (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 is ResNet-50 final channel size
        in_dim = 2048

        # Two-FC projector (mimic your example: Linear -> BN -> ReLU -> Linear)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, feature_dim, bias=True)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_feature(self, x):
        """
        Args:
            return_features (bool): return the projected feature (L2-normalized if normalize_feats=True)
            return_logits (bool): return classifier logits (requires num_classes not None)
        Returns:
            If both flags False: returns features (default).
            If both True: returns (features, logits).
            If only one True: returns that item.
        """
        # Backbone
        x = self.stem(x)
        x = self.avgpool(x)  # (N, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (N, 2048)

        # Project to feature_dim
        feats = self.projector(x)  # (N, feature_dim)
        if self.normalize_feats:
            feats = F.normalize(feats, dim=1)

        return feats

    def forward(self, x):
        out = self.extract_feature(x)
        # Classification output
        out = self.classifier(out)
        return out
