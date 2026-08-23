# -*- coding: utf-8 -*-
"""
Multi-Scale Attention Refined GAB

This replaces the previous GAB module.

Input : [B, C, H, W]
Output: [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = self.sigmoid(attn)

        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)

        return x * attn


class MultiScaleRefinement(nn.Module):
    """
    Multi-scale local refinement:
    3x3 branch  : fine lesion boundary details
    5x5 branch  : medium contextual texture
    dilated 3x3 : larger contextual region
    """

    def __init__(self, channels):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        xd = self.branch_dilated(x)

        out = torch.cat([x3, x5, xd], dim=1)
        out = self.fuse(out)

        return out


class GAB(nn.Module):
    """
    Multi-Scale Attention Refined Global Attention Block.

    Improvements over original GAB:
    1. Global attention captures long-range dependency.
    2. Multi-scale refinement captures local lesion texture and boundary.
    3. Channel attention suppresses irrelevant feature channels.
    4. Spatial attention suppresses false-positive background regions.
    5. Residual fusion stabilizes training.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        # Query projection
        self.query_proj = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # Key projection
        self.key_proj = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # Value projection
        self.value_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Global attention output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

        # New multi-scale refinement
        self.multi_scale = MultiScaleRefinement(channels)

        # New channel-spatial refinement
        self.channel_attn = ChannelAttention(channels, reduction=8)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.gamma_global = nn.Parameter(torch.zeros(1))
        self.gamma_local = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        B, C, H, W = x.shape
        N = H * W

        residual = x

        # =================================================
        # Global attention pathway
        # =================================================
        query = self.query_proj(x)
        key = self.key_proj(x)

        query = query.view(B, -1, N)
        key = key.view(B, -1, N)

        attention = torch.bmm(
            query.permute(0, 2, 1),
            key
        )

        attention = F.softmax(attention, dim=-1)

        value = self.value_proj(x)
        value = value.view(B, C, N)

        global_out = torch.bmm(
            value,
            attention.permute(0, 2, 1)
        )

        global_out = global_out.view(B, C, H, W)
        global_out = self.out_proj(global_out)

        # =================================================
        # Multi-scale local refinement pathway
        # =================================================
        local_out = self.multi_scale(x)
        local_out = self.channel_attn(local_out)
        local_out = self.spatial_attn(local_out)

        # =================================================
        # Fusion
        # =================================================
        fused = torch.cat([
            self.gamma_global * global_out,
            self.gamma_local * local_out
        ], dim=1)

        fused = self.final_fusion(fused)

        out = fused + residual

        return out
"""

if __name__ == "__main__":
    x = torch.randn(2, 64, 128, 128)

    gab = GAB(channels=64, reduction=4)

    y = gab(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
    """