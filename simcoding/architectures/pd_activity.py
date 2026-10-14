"""Lightweight Transformer for PD Activity representations."""
import torch
from torch import nn
from torch.nn import functional as F


class PDActivityTransformer(nn.Module):
    """Map activity tensors [batch, 600, 384] to unit-norm 128-D features.

    Defaults use 25 temporal patches of 24 x 384 values, a single encoder
    layer with four attention heads, and a 64-D feedforward hidden layer.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=1,
                 dim_feedforward=64, dropout=0.0, patch_time=24,
                 feature_dim=384, num_patches=25):
        super().__init__()
        self.patch_time = patch_time
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.patch_embed = nn.Linear(patch_time * feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_fc1 = nn.Linear(d_model, d_model)
        for module in (self.patch_embed, self.cls_fc1):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        expected = (self.patch_time * self.num_patches, self.feature_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected[0]}, {expected[1]}]")
        patches = x.reshape(x.shape[0], self.num_patches, -1)
        encoded = self.encoder(self.patch_embed(patches) + self.pos_embedding)
        return F.normalize(self.cls_fc1(encoded.mean(dim=1)), dim=-1)
