# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:17:16 2026

@author: USER
"""
import torch
import torch.nn as nn
from GAB import GAB

class SkipGABFusion(nn.Module):
    def __init__(self, enc_channels, dec_channels, out_channels):
        super().__init__()

        self.enc_proj = nn.Conv2d(enc_channels, out_channels, kernel_size=1)
        self.dec_proj = nn.Conv2d(dec_channels, out_channels, kernel_size=1)

        self.gab = GAB(out_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        enc_feat = self.enc_proj(enc_feat)
        dec_feat = self.dec_proj(dec_feat)

        enc_feat = self.gab(enc_feat)

        fused = torch.cat([enc_feat, dec_feat], dim=1)
        fused = self.fusion(fused)

        return fused