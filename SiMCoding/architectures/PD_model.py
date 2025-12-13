import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class PD_Merged_Transformer(nn.Module):
    """
    输入: x [B, 600, 384]
    patch_time = 24 → num_patches = 25
    每个 patch: 24 × 384 = 9216 维，映射到 d_model 维，然后做 Transformer 编码。
    输出: 每个样本的 embedding 向量 [B, d_model]
    """

    def __init__(
            self,
            num_classes: int = None,  # 如果只做特征提取，可以不用
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 1,
            dim_feedforward: int = 64,
            dropout: float = 0.0,
            patch_time: int = 24,
            feature_dim: int = 384,
            num_patches: int = 25,
    ):
        super().__init__()

        self.patch_time = patch_time
        self.feature_dim = feature_dim
        self.num_patches = num_patches

        patch_dim = patch_time * feature_dim  # 24 * 384 = 9216

        # 1) patch embedding: [B, N, patch_dim] → [B, N, d_model]
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # 2) 可学习的位置编码 [1, N, d_model]
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 3) Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) 一个小的投影头（可选）
        self.cls_fc1 = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        if self.cls_fc1.bias is not None:
            nn.init.zeros_(self.cls_fc1.bias)

    def forward(self, x):
        """
        x: [B, 600, 384]
        返回: [B, d_model] 的 embedding（给 RoughLearning / Logistic Regression 用）
        """
        B, T, C = x.shape
        assert C == self.feature_dim, f"输入特征维 C={C}, 期望={self.feature_dim}"
        assert T == self.patch_time * self.num_patches, \
            f"T={T} 不等于 patch_time*num_patches={self.patch_time * self.num_patches}"

        # [B, 600, 384] → [B, 25, 24, 384]
        x = x.view(B, self.num_patches, self.patch_time, C)

        # → [B, 25, 9216]
        x = x.reshape(B, self.num_patches, -1)

        # patch + pos
        x = self.patch_embed(x)  # [B, 25, d_model]
        x = x + self.pos_embedding

        # Transformer 编码
        x = self.encoder(x)  # [B, 25, d_model]

        # 对所有 patch 做平均池化
        x = x.mean(dim=1)  # [B, d_model]

        # 小投影 + 归一化（适合对比学习 / RoughLearning）
        x = self.cls_fc1(x)
        x = F.normalize(x, dim=-1)

        return x


class PD_Merged_Transformer_CE(nn.Module):
    """
    与 PDTransformer 结构类似，但输出分类 logits，用于 CE 监督训练。

    输入: x [B, 600, 384]
    输出: logits [B, num_classes]
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 1,
            dim_feedforward: int = 64,
            dropout: float = 0.0,
            patch_time: int = 24,
            feature_dim: int = 384,
            num_patches: int = 25,
    ):
        super().__init__()

        self.patch_time = patch_time
        self.feature_dim = feature_dim
        self.num_patches = num_patches

        patch_dim = patch_time * feature_dim  # 24 * 384 = 9216

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 分类头：先投影，再输出 logits
        self.cls_fc1 = nn.Linear(d_model, d_model)
        self.cls_fc2 = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        if self.cls_fc1.bias is not None:
            nn.init.zeros_(self.cls_fc1.bias)

        nn.init.xavier_uniform_(self.cls_fc2.weight)
        if self.cls_fc2.bias is not None:
            nn.init.zeros_(self.cls_fc2.bias)

    def forward(self, x):
        """
        x: [B, 600, 384]
        返回:
          logits: [B, num_classes]
        """
        B, T, C = x.shape
        assert C == self.feature_dim, f"输入特征维 C={C}, 期望={self.feature_dim}"
        assert T == self.patch_time * self.num_patches, \
            f"T={T} 不等于 patch_time*num_patches={self.patch_time * self.num_patches}"

        # [B, 600, 384] → [B, 25, 24, 384]
        x = x.view(B, self.num_patches, self.patch_time, C)

        # → [B, 25, 9216]
        x = x.reshape(B, self.num_patches, -1)

        # patch + pos
        x = self.patch_embed(x)  # [B, 25, d_model]
        x = x + self.pos_embedding

        # Transformer 编码
        x = self.encoder(x)  # [B, 25, d_model]

        # 对所有 patch 做平均池化
        x = x.mean(dim=1)  # [B, d_model]

        # 分类头
        x = self.cls_fc1(x)
        x = F.normalize(x, dim=-1)  # 保持和你之前习惯一致
        logits = self.cls_fc2(x)  # [B, num_classes]

        return logits


class PDTransformer_CE(nn.Module):
    """
    轻量级 Transformer，用于处理每个样本 600×12 的传感器数据。

    流程：
      输入: x, 形状 [B, 600, 12]
      1) 切成 50 个 patch，每个 patch 大小 12×12
      2) 每个 patch 展平为 144 维向量
      3) 线性映射到 d_model 维度 + 位置编码
      4) 送入 TransformerEncoder
      5) 对所有 patch 的输出做平均池化（或可以改成 CLS token）
      6) 全连接层输出类别 logits
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.0,
            patch_time: int = 24,
            feature_dim: int = 24,
            num_patches: int = 25,
    ):
        super().__init__()

        self.patch_time = patch_time  # 每个 patch 时间长度 = 12
        self.feature_dim = feature_dim  # 每个时间点的特征维度 = 12
        self.num_patches = num_patches  # patch 数量 = 50

        patch_dim = patch_time * feature_dim  # 12 * 12 = 144

        # 1) patch embedding: [B, N, patch_dim] → [B, N, d_model]
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # 2) 可学习的位置编码，[1, N, d_model]
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, d_model)
        )

        # 3) Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 [B, N, d_model] 的输入格式
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) 分类头：对 pooled 的特征做线性分类
        self.cls_fc1 = nn.Linear(d_model, d_model)
        self.cls_fc2 = nn.Linear(d_model, num_classes)

        # 可选初始化
        self._init_weights()

    def _init_weights(self):
        # 简单初始化
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        if self.cls_fc1.bias is not None:
            nn.init.zeros_(self.cls_fc1.bias)

        nn.init.xavier_uniform_(self.cls_fc2.weight)
        if self.cls_fc2.bias is not None:
            nn.init.zeros_(self.cls_fc2.bias)

    def forward(self, x):
        """
        x: [B, 600, 12]  (time, feature)

        返回:
          logits: [B, num_classes]
        """
        B, T, C = x.shape
        assert T == self.patch_time * self.num_patches, \
            f"输入时间长度 T={T} 不等于 patch_time*num_patches={self.patch_time * self.num_patches}"

        # 1) 切 patch
        # 先 reshape: [B, 600, 12] → [B, num_patches, patch_time, C]
        x = x.view(B, self.num_patches, self.patch_time, C)  # [B, 50, 12, 12]

        # 2) 每个 patch 展平为向量: [B, 50, 12*12=144]
        x = x.reshape(B, self.num_patches, -1)

        # 3) 线性映射到 d_model
        x = self.patch_embed(x)  # [B, 50, d_model]

        # 4) 加位置编码
        x = x + self.pos_embedding  # [B, 50, d_model]

        # 5) 送入 Transformer Encoder
        x = self.encoder(x)  # [B, 50, d_model]

        # 6) 池化：对所有 patch 求平均，得到一个全局表示
        # 也可以改成 x[:, 0, :] 用 CLS token 的方式
        x = x.mean(dim=1)  # [B, d_model]

        # 7) 分类头
        x = self.cls_fc1(x)
        x = F.normalize(x)
        logits = self.cls_fc2(x)  # [B, num_classes]

        return logits


class PDTransformer(nn.Module):
    """
    轻量级 Transformer，用于处理每个样本 600×12 的传感器数据。

    流程：
      输入: x, 形状 [B, 600, 12]
      1) 切成 50 个 patch，每个 patch 大小 12×12
      2) 每个 patch 展平为 144 维向量
      3) 线性映射到 d_model 维度 + 位置编码
      4) 送入 TransformerEncoder
      5) 对所有 patch 的输出做平均池化（或可以改成 CLS token）
      6) 全连接层输出类别 logits
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.0,
            patch_time: int = 24,
            feature_dim: int = 24,
            num_patches: int = 25,
    ):
        super().__init__()

        self.patch_time = patch_time  # 每个 patch 时间长度 = 12
        self.feature_dim = feature_dim  # 每个时间点的特征维度 = 12
        self.num_patches = num_patches  # patch 数量 = 50

        patch_dim = patch_time * feature_dim  # 12 * 12 = 144

        # 1) patch embedding: [B, N, patch_dim] → [B, N, d_model]
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # 2) 可学习的位置编码，[1, N, d_model]
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, d_model)
        )

        # 3) Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使用 [B, N, d_model] 的输入格式
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) 分类头：对 pooled 的特征做线性分类
        self.cls_fc1 = nn.Linear(d_model, d_model)
        self.cls_fc2 = nn.Linear(d_model, num_classes)

        # 可选初始化
        self._init_weights()

    def _init_weights(self):
        # 简单初始化
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        if self.cls_fc1.bias is not None:
            nn.init.zeros_(self.cls_fc1.bias)

        nn.init.xavier_uniform_(self.cls_fc2.weight)
        if self.cls_fc2.bias is not None:
            nn.init.zeros_(self.cls_fc2.bias)

    def forward(self, x):
        """
        x: [B, 600, 12]  (time, feature)

        返回:
          logits: [B, num_classes]
        """
        B, T, C = x.shape
        assert T == self.patch_time * self.num_patches, \
            f"输入时间长度 T={T} 不等于 patch_time*num_patches={self.patch_time * self.num_patches}"

        # 1) 切 patch
        # 先 reshape: [B, 600, 12] → [B, num_patches, patch_time, C]
        x = x.view(B, self.num_patches, self.patch_time, C)  # [B, 50, 12, 12]

        # 2) 每个 patch 展平为向量: [B, 50, 12*12=144]
        x = x.reshape(B, self.num_patches, -1)

        # 3) 线性映射到 d_model
        x = self.patch_embed(x)  # [B, 50, d_model]

        # 4) 加位置编码
        x = x + self.pos_embedding  # [B, 50, d_model]

        # 5) 送入 Transformer Encoder
        x = self.encoder(x)  # [B, 50, d_model]

        # 6) 池化：对所有 patch 求平均，得到一个全局表示
        # 也可以改成 x[:, 0, :] 用 CLS token 的方式
        x = x.mean(dim=1)  # [B, d_model]

        # 7) 分类头
        x = self.cls_fc1(x)
        x = F.normalize(x)
        return x


# -----------------------------------------------------
# Define MLP model
# -----------------------------------------------------
class PD_Speech(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

            # nn.ReLU(),
            # nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x)
        return x


class PD_Speech_CE(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x)
        x = self.fc(x)
        return x
