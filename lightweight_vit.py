# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:20:21 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Efficient MLP
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Lightweight Linear Attention
# ---------------------------------------------------------
class LinearAttention(nn.Module):
    """
    Lightweight linear-complexity attention.

    Input : [B, N, C]
    Output: [B, N, C]
    """

    def __init__(self, dim, num_heads=4, qkv_bias=True, dropout=0.0):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        kv = torch.matmul(k.transpose(-2, -1), v)

        z = 1.0 / (
            torch.matmul(q, k.sum(dim=-2).unsqueeze(-1)) + 1e-6
        )

        out = torch.matmul(q, kv) * z

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


# ---------------------------------------------------------
# Lightweight Transformer Block
# ---------------------------------------------------------
class LightViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    Converts image/features into patch tokens.

    Input : [B, C, H, W]
    Output: tokens [B, H*W, embed_dim], H, W
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size=16
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        return tokens, H, W


# ---------------------------------------------------------
# Patch Merging
# ---------------------------------------------------------
class PatchMerging(nn.Module):
    """
    Downsamples spatial resolution by 2.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.reduction = nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.reduction(x)

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        return tokens, H, W


# ---------------------------------------------------------
# Lightweight Hierarchical ViT Encoder
# ---------------------------------------------------------
class LightViTEncoder(nn.Module):
    """
    Lightweight ViT encoder for the proposed architecture.

    Output:
        f1: [B, C1, H/2,  W/2]
        f2: [B, C2, H/4,  W/4]
        f3: [B, C3, H/8,  W/8]
        f4: [B, C4, H/16, W/16]
    """

    def __init__(
        self,
        in_channels=3,
        dims=(32, 64, 128, 256),
        depths=(2, 2, 2, 2),
        heads=(1, 2, 4, 8),
        mlp_ratio=2.0,
        dropout=0.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=dims[0],
            patch_size=2
        )

        self.stage1 = nn.ModuleList([
            LightViTBlock(
                dim=dims[0],
                num_heads=heads[0],
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depths[0])
        ])

        self.merge1 = PatchMerging(dims[0], dims[1])

        self.stage2 = nn.ModuleList([
            LightViTBlock(
                dim=dims[1],
                num_heads=heads[1],
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depths[1])
        ])

        self.merge2 = PatchMerging(dims[1], dims[2])

        self.stage3 = nn.ModuleList([
            LightViTBlock(
                dim=dims[2],
                num_heads=heads[2],
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depths[2])
        ])

        self.merge3 = PatchMerging(dims[2], dims[3])

        self.stage4 = nn.ModuleList([
            LightViTBlock(
                dim=dims[3],
                num_heads=heads[3],
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depths[3])
        ])

        self.dims = dims

    def tokens_to_feature(self, tokens, H, W):
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, H, W)

    def forward_stage(self, tokens, blocks):
        for block in blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, x):
        # Stage 1
        x, H, W = self.patch_embed(x)
        x = self.forward_stage(x, self.stage1)
        f1 = self.tokens_to_feature(x, H, W)

        # Stage 2
        x, H, W = self.merge1(f1)
        x = self.forward_stage(x, self.stage2)
        f2 = self.tokens_to_feature(x, H, W)

        # Stage 3
        x, H, W = self.merge2(f2)
        x = self.forward_stage(x, self.stage3)
        f3 = self.tokens_to_feature(x, H, W)

        # Stage 4
        x, H, W = self.merge3(f3)
        x = self.forward_stage(x, self.stage4)
        f4 = self.tokens_to_feature(x, H, W)

        return [f1, f2, f3, f4]
"""    
if __name__ == "__main__":

    x = torch.randn(2, 3, 256, 256)

    encoder = LightViTEncoder(
        in_channels=3,
        dims=(32, 64, 128, 256),
        depths=(2, 2, 2, 2),
        heads=(1, 2, 4, 8),
        dropout=0.1
    )

    features = encoder(x)

    for i, f in enumerate(features):
        print(f"Feature {i+1}: {f.shape}")    
"""        