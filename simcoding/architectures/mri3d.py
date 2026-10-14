"""Volumetric ResNet-18 for SiMCoding MRI representations."""
import torch
from torch import nn
from torch.nn import functional as F


class _DeterministicMaxPool3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output, indices = F.max_pool3d(x, 3, 2, 1, return_indices=True)
        ctx.save_for_backward(indices)
        ctx.input_shape = x.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # PyTorch's deterministic CUDA scatter_add accumulates overlapping
        # windows without the unsupported max_pool3d CUDA backward kernel.
        grad_input = grad_output.new_zeros(ctx.input_shape)
        grad_input.flatten(2).scatter_add_(2, indices.flatten(2), grad_output.flatten(2))
        return grad_input


class DeterministicStemMaxPool3d(nn.MaxPool3d):
    def __init__(self):
        super().__init__(3, 2, 1)

    def forward(self, x):
        if torch.are_deterministic_algorithms_enabled():
            return _DeterministicMaxPool3d.apply(x)
        return super().forward(x)


class DeterministicGlobalAvgPool3d(nn.AdaptiveAvgPool3d):
    def __init__(self):
        super().__init__(1)

    def forward(self, x):
        if torch.are_deterministic_algorithms_enabled():
            return x.mean(dim=(2, 3, 4), keepdim=True)
        return super().forward(x)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(h)) + self.shortcut(x), inplace=True)


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                  DeterministicStemMaxPool3d())
        stages, cin = [], 64
        for i, cout in enumerate((64, 128, 256, 512)):
            stages.append(nn.Sequential(BasicBlock(cin, cout, 1 if i == 0 else 2),
                                        BasicBlock(cout, cout)))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.pool = DeterministicGlobalAvgPool3d()
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.pool(self.stages(self.stem(x))).flatten(1)


class MRI3DResNet18(nn.Module):
    """Map single-channel volumes [batch, 1, depth, height, width] to features.

    A 3D ResNet-18 produces 512 pooled features, followed by a linear
    projection to 128 dimensions and L2 normalization.
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projection = nn.Linear(512, feature_dim)

    def features(self, x):
        z = self.projection(self.encoder(x))
        return F.normalize(z.float(), p=2, dim=1)

    def forward(self, x):
        return self.features(x)
