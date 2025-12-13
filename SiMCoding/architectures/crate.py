import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init


def pair(t):
    # ensure a value is a (h, w) tuple
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    # LayerNorm -> fn(x): a common pattern that stabilizes training
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # normalize first, then apply the wrapped sublayer (attention/ff)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    ISTA-like proximal update block.
    Interprets a learned dictionary D (= self.weight^T) and performs
    one gradient/prox step: x <- ReLU( x + η (D^T x - D^T D x) - η λ ).
    """

    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        # learnable 'dictionary' weight (dim x dim); symmetric update uses both W and W^T
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)  # good default init for linear weights
        self.step_size = step_size  # η in the paper; size of the gradient step
        self.lambd = 0.1  # ℓ1-like shrinkage (soft-threshold surrogate)

    def forward(self, x):
        # compute D x  (not used directly) then D^T D x
        x1 = F.linear(x, self.weight, bias=None)  # x @ W^T
        grad_1 = F.linear(x1, self.weight.t(), bias=None)  # (x @ W^T) @ W = x (W^T W)
        # compute D^T x
        grad_2 = F.linear(x, self.weight.t(), bias=None)  # x @ W
        # gradient update toward sparse coding optimum: η (D^T x - D^T D x) - η λ
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd  # introduce sparsity
        # ReLU acts like a proximal operator for nonnegativity / sparsity (soft shrink surrogate)
        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    """
    MSSA-style attention approximation.
    NOTE: This uses a *single* projection (qkv) -> w and then forms dots = w w^T,
    i.e., self-similarity in the head subspace. This differs from standard Q,K,V.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # per-head dims × heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # scale to stabilize softmax

        self.attend = nn.Softmax(dim=-1)  # attention along token dimension
        self.dropout = nn.Dropout(dropout)

        # single linear maps tokens -> head space; (not the usual 3*dim projection)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        # output projection back to model dim (or identity if shape matches)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # project to head space: shape (b, n, h*dim_head) -> (b, h, n, dim_head)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        # pairwise token similarity per head: (b,h,n,d) @ (b,h,d,n) -> (b,h,n,n)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        # attention weights over tokens
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # store for visualization
        self.last_attn = attn.detach()

        # weighted combination in the same head subspace: (b,h,n,n) @ (b,h,n,d) -> (b,h,n,d)
        out = torch.matmul(attn, w)

        # merge heads back: (b,h,n,d) -> (b,n,h*d)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # project to model dimension


class Transformer(nn.Module):
    # Stack of (PreNorm(Attention), PreNorm(FeedForward)) blocks — i.e., MSSA + ISTA per layer
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # MSSA-like compression step
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        # ISTA-like sparsification step with step size 'ista'
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

    def forward(self, x):
        depth = 0  # (unused counter)
        for attn, ff in self.layers:
            # residual after attention (gradient-like compression step)
            grad_x = attn(x) + x
            # proximal / shrinkage step (sparsification)
            x = ff(grad_x)
        return x


class CRATE(nn.Module):
    """
    White-box ViT:
      - Patch embed -> tokens
      - +CLS token and positional encodings
      - Transformer layers ≈ alternating compression (MSSA) and sparsification (ISTA)
      - Pool (CLS or mean) -> classifier head
    """

    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            dropout=0., emb_dropout=0., ista=0.1
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # sanity: images must tile perfectly into patches
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Flatten each patch -> linear project to model dim (with LayerNorms around)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # positional embedding and learned [CLS] token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # transformer = stack of white-box blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)

        self.pool = pool
        self.to_latent = nn.Identity()  # placeholder if you later add a projection

        # classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # def forward(self, img):
    #     # (B,C,H,W) -> (B,N,dim)
    #     x = self.to_patch_embedding(img)
    #     b, n, _ = x.shape
    #
    #     # prepend [CLS] token and add 2D positional encodings
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)
    #     x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x)
    #
    #     # white-box transformer layers
    #     x = self.transformer(x)
    #     feature_pre = x  # retained features (not used in return)
    #
    #     # pool: CLS token or mean over tokens
    #     x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
    #
    #     x = self.to_latent(x)
    #     feature_last = x  # retained features (not used in return)
    #     return self.mlp_head(x)  # logits

    def forward(self, img, return_features=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        if return_features:
            return x  # features before classification head

        return self.mlp_head(x)


def CRATE_tiny(num_classes=1000):
    # ~ViT-Tiny scale, heads chosen so dim_head = dim / heads
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6
    )


def CRATE_small(num_classes=1000):
    # ~ViT-Small scale
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12
    )


def CRATE_base(num_classes=1000):
    # ~ViT-Base scale
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12
    )


def CRATE_large(num_classes=1000):
    # larger model: deeper and wider
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16
    )


def CRATE_tiny_mnist(num_classes=10):
    # MNIST is 28x28 grayscale; pick a light model to avoid overfitting
    # Option A (faster): patch_size=7 → 16 tokens
    return CRATE(
        image_size=28,
        patch_size=7,  # 28 % 7 == 0 → 4x4 = 16 tokens
        num_classes=num_classes,  # 10 classes for MNIST
        dim=64,  # model width
        depth=6,  # number of layers
        heads=4,  # attention heads
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64 // 4,  # 32
        channels=1
        # IMPORTANT: grayscale input
        # we must pass channels=1 so the patch embed expects a single channel
        # (add this parameter below — the CRATE __init__ already supports it)
    )


def CRATE_tiny_cifar(num_classes=10):
    return CRATE(
        image_size=32,
        patch_size=8,  # 32 % 8 == 0 → 4x4 = 16 tokens
        num_classes=num_classes,  # 10 classes for MNIST
        dim=512,  # model width
        depth=8,  # number of layers
        heads=8,  # attention heads
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=512 // 8,  # 64
        channels=3
    )


def CRATE_tiny_imagenette(num_classes=10):
    return CRATE(
        image_size=128,
        patch_size=8,  # 64 tokens → healthy for 32×32
        num_classes=num_classes,
        dim=512,
        depth=8,
        heads=8,
        dim_head=512 // 8,  # 32
        dropout=0.0,
        emb_dropout=0.0,
        channels=3
    )


def CRATE_AD(num_classes=2):
    return CRATE(
        image_size=224,
        patch_size=16,  # 14*14
        num_classes=num_classes,
        dim=512,
        depth=12,
        heads=8,
        dim_head=512 // 8,  # 32
        dropout=0.0,
        emb_dropout=0.0,
        channels=1
    )


def CRATE_PD_total(num_classes=2,
                   image_size=(600, 384),
                   patch_size=(24, 384),  # or (12, 12) if you prefer 2D patches
                   dim=512,
                   depth=12,
                   heads=4):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        channels=1,  # PD data: single "channel" matrix
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=dim // heads,
        ista=0.1
    )


def CRATE_PD_subset(num_classes=2,
                    image_size=(600, 24),
                    patch_size=(24, 24),  # or (12, 12) if you prefer 2D patches
                    dim=512,
                    depth=12,
                    heads=4):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        channels=1,  # PD data: single "channel" matrix
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=dim // heads,
        ista=0.1
    )


def CRATE_PD_Speech(num_classes=2,
                    image_size=(1, 752),
                    patch_size=(1, 47),  # or (12, 12) if you prefer 2D patches
                    dim=64,
                    depth=8,
                    heads=4):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        channels=1,  # PD data: single "channel" matrix
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=dim // heads,
        ista=0.1
    )
