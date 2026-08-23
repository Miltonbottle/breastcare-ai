# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:59:21 2026

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Dice Loss
# =========================================================
class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    logits : [B, 1, H, W]
    targets: [B, 1, H, W]
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        loss = 1.0 - dice

        return loss.mean()


# =========================================================
# BCE + Dice Loss
# =========================================================
class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and DiceLoss.

    Recommended for breast lesion segmentation.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return loss


# =========================================================
# Tversky Loss
# =========================================================
class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced lesion segmentation.

    alpha controls false positives.
    beta controls false negatives.

    For medical segmentation:
    beta > alpha gives stronger penalty to missed lesions.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = 1.0 - tversky

        return loss.mean()


# =========================================================
# Focal Tversky Loss
# =========================================================
class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss.

    Useful for small and difficult breast lesion segmentation.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()

        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )

        self.gamma = gamma

    def forward(self, logits, targets):
        tversky_loss = self.tversky(logits, targets)

        focal_tversky_loss = torch.pow(tversky_loss, self.gamma)

        return focal_tversky_loss


# =========================================================
# BCE + Focal Tversky Loss
# =========================================================
class BCEFocalTverskyLoss(nn.Module):
    """
    BCE + Focal Tversky Loss.

    Strong option for small datasets and imbalanced masks.
    """

    def __init__(
        self,
        bce_weight=0.4,
        focal_tversky_weight=0.6,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        smooth=1e-6
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_tversky_weight = focal_tversky_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth
        )

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        ft_loss = self.focal_tversky(logits, targets)

        loss = (
            self.bce_weight * bce_loss
            + self.focal_tversky_weight * ft_loss
        )

        return loss


# =========================================================
# Boundary Loss
# =========================================================
class BoundaryLoss(nn.Module):
    """
    Lightweight boundary-aware loss using Sobel gradients.

    Encourages accurate lesion boundary segmentation.
    """

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_edges(self, x):
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edge

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        pred_edges = self.get_edges(probs)
        target_edges = self.get_edges(targets)

        loss = F.l1_loss(pred_edges, target_edges)

        return loss


# =========================================================
# Final Recommended Loss
# =========================================================
class BreastLesionSegmentationLoss(nn.Module):
    """
    Final recommended loss for breast ultrasound lesion segmentation.

    Combines:
    1. BCE loss
    2. Dice loss
    3. Focal Tversky loss
    4. Boundary loss

    Suitable for:
    - small lesion regions
    - blurred boundaries
    - class imbalance
    - limited dataset size
    """

    def __init__(
        self,
        bce_weight=0.3,
        dice_weight=0.3,
        focal_tversky_weight=0.3,
        boundary_weight=0.1,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        smooth=1e-6
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.boundary_weight = boundary_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth
        )

        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        ft_loss = self.focal_tversky(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.focal_tversky_weight * ft_loss
            + self.boundary_weight * boundary_loss
        )

        return total_loss


# =========================================================
# Usage Example
# =========================================================
if __name__ == "__main__":

    criterion = BreastLesionSegmentationLoss(
        bce_weight=0.3,
        dice_weight=0.3,
        focal_tversky_weight=0.3,
        boundary_weight=0.1
    )

    logits = torch.randn(2, 1, 256, 256)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float()

    loss = criterion(logits, masks)

    print("Loss:", loss.item())