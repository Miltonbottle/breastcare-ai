# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:58:09 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Utility: Channel Shuffle
# ---------------------------------------------------------
class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        assert C % g == 0

        x = x.view(B, g, C // g, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x


# ---------------------------------------------------------
# Selective Scan Core: Simplified S6 Block
# ---------------------------------------------------------
class SelectiveScan1D(nn.Module):
    """
    Simplified selective state-space scan used inside SS2D.
    Input:  x -> [B, L, C]
    Output: y -> [B, L, C]
    """

    def __init__(self, dim, state_dim=16):
        super().__init__()

        self.dim = dim
        self.state_dim = state_dim

        self.x_proj = nn.Linear(dim, dim * 3)
        self.dt_proj = nn.Linear(dim, dim)

        self.A_log = nn.Parameter(torch.randn(dim, state_dim))
        self.D = nn.Parameter(torch.ones(dim))

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape

        x_proj = self.x_proj(x)
        delta, B_param, C_param = torch.chunk(x_proj, 3, dim=-1)

        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log)          # [C, N]
        D = self.D                         # [C]

        h = torch.zeros(B, C, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            xt = x[:, t]                   # [B, C]
            dt = delta[:, t]               # [B, C]
            Bt = B_param[:, t]             # [B, C]
            Ct = C_param[:, t]             # [B, C]

            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * Bt.unsqueeze(-1)

            h = dA * h + dB * xt.unsqueeze(-1)

            yt = torch.sum(h * Ct.unsqueeze(-1), dim=-1)
            yt = yt + D.unsqueeze(0) * xt

            ys.append(yt)

        y = torch.stack(ys, dim=1)
        y = self.out_proj(y)

        return y


# ---------------------------------------------------------
# SS2D: Vision Mamba 2D Selective Scan
# ---------------------------------------------------------
class SS2D(nn.Module):
    """
    Vision Mamba SS2D block.

    Performs four directional scans:
    1. left-to-right
    2. right-to-left
    3. top-to-bottom
    4. bottom-to-top
    """

    def __init__(self, dim, state_dim=16):
        super().__init__()

        self.scan_lr = SelectiveScan1D(dim, state_dim)
        self.scan_rl = SelectiveScan1D(dim, state_dim)
        self.scan_tb = SelectiveScan1D(dim, state_dim)
        self.scan_bt = SelectiveScan1D(dim, state_dim)

        self.merge = nn.Linear(dim * 4, dim)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """

        B, H, W, C = x.shape

        # -------------------------------
        # Horizontal scan: left-to-right
        # -------------------------------
        x_lr = x.reshape(B * H, W, C)
        y_lr = self.scan_lr(x_lr)
        y_lr = y_lr.reshape(B, H, W, C)

        # -------------------------------
        # Horizontal scan: right-to-left
        # -------------------------------
        x_rl = torch.flip(x, dims=[2]).reshape(B * H, W, C)
        y_rl = self.scan_rl(x_rl)
        y_rl = y_rl.reshape(B, H, W, C)
        y_rl = torch.flip(y_rl, dims=[2])

        # -------------------------------
        # Vertical scan: top-to-bottom
        # -------------------------------
        x_tb = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_tb = self.scan_tb(x_tb)
        y_tb = y_tb.reshape(B, W, H, C).permute(0, 2, 1, 3)

        # -------------------------------
        # Vertical scan: bottom-to-top
        # -------------------------------
        x_bt = torch.flip(x, dims=[1])
        x_bt = x_bt.permute(0, 2, 1, 3).reshape(B * W, H, C)

        y_bt = self.scan_bt(x_bt)
        y_bt = y_bt.reshape(B, W, H, C).permute(0, 2, 1, 3)
        y_bt = torch.flip(y_bt, dims=[1])

        # -------------------------------
        # Scan merging
        # -------------------------------
        y = torch.cat([y_lr, y_rl, y_tb, y_bt], dim=-1)
        y = self.merge(y)

        return y


# ---------------------------------------------------------
# VSSM Block
# ---------------------------------------------------------
class VSSMBlock(nn.Module):
    """
    VSSM block as shown in the figure.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        state_dim=16,
        expansion=2,
        dropout=0.0
    ):
        super().__init__()

        hidden_dim = dim // 2
        expanded_dim = hidden_dim * expansion

        # Channel split branch 1
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # Branch 2: VSSM / SS2D pathway
        self.norm = nn.LayerNorm(hidden_dim)

        self.in_proj = nn.Linear(hidden_dim, expanded_dim * 2)

        self.dwconv = nn.Conv2d(
            expanded_dim,
            expanded_dim,
            kernel_size=3,
            padding=1,
            groups=expanded_dim
        )

        self.ss2d = SS2D(expanded_dim, state_dim)

        self.norm_after_ss2d = nn.LayerNorm(expanded_dim)

        self.out_proj = nn.Linear(expanded_dim, hidden_dim)

        self.shuffle = ChannelShuffle(groups=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        residual = x

        B, C, H, W = x.shape
        assert C % 2 == 0, "Channel dimension must be divisible by 2."

        # -------------------------------------------------
        # Channel split
        # -------------------------------------------------
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # -------------------------------------------------
        # Upper convolutional branch
        # -------------------------------------------------
        y1 = self.branch1(x1)

        # -------------------------------------------------
        # Lower SS2D branch
        # -------------------------------------------------
        x2 = x2.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C/2]

        x2 = self.norm(x2)

        x_proj = self.in_proj(x2)
        x_ssm, gate = torch.chunk(x_proj, chunks=2, dim=-1)

        # depthwise convolution
        x_ssm = x_ssm.permute(0, 3, 1, 2).contiguous()
        x_ssm = self.dwconv(x_ssm)
        x_ssm = x_ssm.permute(0, 2, 3, 1).contiguous()

        # SS2D directional selective scan
        x_ssm = self.ss2d(x_ssm)

        x_ssm = self.norm_after_ss2d(x_ssm)

        # gating / multiplication
        x_ssm = x_ssm * F.silu(gate)

        y2 = self.out_proj(x_ssm)
        y2 = self.dropout(y2)

        y2 = y2.permute(0, 3, 1, 2).contiguous()

        # -------------------------------------------------
        # Concatenate + channel shuffle + residual add
        # -------------------------------------------------
        y = torch.cat([y1, y2], dim=1)

        y = self.shuffle(y)

        y = y + residual

        return y

"""
# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
if __name__ == "__main__":

    x = torch.randn(2, 64, 128, 128)

    block = VSSMBlock(
        dim=64,
        state_dim=16,
        expansion=2,
        dropout=0.1
    )

    y = block(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
    
"""    