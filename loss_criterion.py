# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:00:11 2026

@author: USER
"""

criterion = BreastLesionSegmentationLoss(
    bce_weight=0.3,
    dice_weight=0.3,
    focal_tversky_weight=0.3,
    boundary_weight=0.1,
    alpha=0.3,
    beta=0.7,
    gamma=0.75
)

criterion = BCEDiceLoss(
    bce_weight=0.5,
    dice_weight=0.5
)