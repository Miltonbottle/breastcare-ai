# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:38:07 2026

@author: USER
"""



import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

from hybridsegnet import HybridViTGABVSSMUNet


# =========================================================
# Configuration
# =========================================================

class CFG:
    test_image_dir = r"C:\Users\USER\Desktop\Breast_cancer\Dataset\BUSI\images"
    test_mask_dir = r"C:\Users\USER\Desktop\Breast_cancer\Dataset\BUSI\masks"

    checkpoint_path = "checkpoints/fold_1_best_model.pth"

    save_dir = "test_results_mal"
    vis_dir = "test_results_mal/visualizations"

    image_size = 256
    batch_size = 1
    num_workers = 0

    threshold = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(CFG.save_dir, exist_ok=True)
os.makedirs(CFG.vis_dir, exist_ok=True)


# =========================================================
# Dataset
# =========================================================

class BreastUltrasoundTestDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        original_image = image.copy()
        original_mask = mask.copy()

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "original_image": original_image,
            "original_mask": original_mask,
            "image_path": image_path
        }


# =========================================================
# Transform
# =========================================================

def get_test_transform():
    return A.Compose([
        A.Resize(CFG.image_size, CFG.image_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


# =========================================================
# Data Paths
# =========================================================

def get_data_paths(image_dir, mask_dir):
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    image_paths = []
    mask_paths = []

    for file_name in image_files:
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        if os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f"Warning: mask not found for {file_name}")

    return image_paths, mask_paths


# =========================================================
# Model
# =========================================================

def build_model():
    model = HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(1, 1, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(pred, target, smooth=1e-6):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    sensitivity = recall
    specificity = (tn + smooth) / (tn + fp + smooth)

    f1 = (2 * precision * recall + smooth) / (
        precision + recall + smooth
    )

    mae = np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32)))

    return {
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "mae": mae,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# =========================================================
# Visualization
# =========================================================

def save_visualization(
    image,
    mask,
    pred,
    save_path,
    alpha=0.45
):
    image = cv2.resize(image, (CFG.image_size, CFG.image_size))
    mask = cv2.resize(mask, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_NEAREST)

    pred = pred.astype(np.uint8)
    mask = mask.astype(np.uint8)

    overlay = image.copy()

    pred_color = np.zeros_like(image)
    pred_color[:, :, 0] = pred * 255

    overlay = cv2.addWeighted(overlay, 1 - alpha, pred_color, alpha, 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Testing
# =========================================================

@torch.no_grad()
def test_model():
    device = torch.device(CFG.device)

    image_paths, mask_paths = get_data_paths(
        CFG.test_image_dir,
        CFG.test_mask_dir
    )

    print("=" * 80)
    print("TESTING MODEL")
    print("=" * 80)
    print(f"Test Images : {len(image_paths)}")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {CFG.checkpoint_path}")
    print("=" * 80)

    test_dataset = BreastUltrasoundTestDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    model = build_model()
    model = load_checkpoint(model, CFG.checkpoint_path, device)
    model = model.to(device)
    model.eval()

    all_metrics = []

    for batch in tqdm(test_loader, desc="Testing"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > CFG.threshold).float()

        for i in range(images.size(0)):
            image_path = batch["image_path"][i]
            file_name = os.path.basename(image_path)

            pred_np = preds[i, 0].detach().cpu().numpy()
            mask_np = masks[i, 0].detach().cpu().numpy()

            pred_np = (pred_np > 0.5).astype(np.uint8)
            mask_np = (mask_np > 0.5).astype(np.uint8)

            metrics = calculate_metrics(pred_np, mask_np)
            metrics["image_name"] = file_name

            all_metrics.append(metrics)

            original_image = batch["original_image"][i].numpy()
            original_mask = batch["original_mask"][i].numpy()

            vis_save_path = os.path.join(
                CFG.vis_dir,
                file_name.replace(".", "_result.")
            )

            save_visualization(
                image=original_image,
                mask=original_mask,
                pred=pred_np,
                save_path=vis_save_path
            )

    results_df = pd.DataFrame(all_metrics)

    per_image_csv = os.path.join(CFG.save_dir, "test_per_image_metrics.csv")
    results_df.to_csv(per_image_csv, index=False)

    mean_results = results_df.drop(columns=["image_name"]).mean()
    std_results = results_df.drop(columns=["image_name"]).std()

    summary = {}

    for metric_name in mean_results.index:
        summary[f"{metric_name}_mean"] = mean_results[metric_name]
        summary[f"{metric_name}_std"] = std_results[metric_name]

    summary_df = pd.DataFrame([summary])

    summary_csv = os.path.join(CFG.save_dir, "test_summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    print(f"Dice        : {summary['dice_mean']:.6f} ± {summary['dice_std']:.6f}")
    print(f"IoU         : {summary['iou_mean']:.6f} ± {summary['iou_std']:.6f}")
    print(f"Accuracy    : {summary['accuracy_mean']:.6f} ± {summary['accuracy_std']:.6f}")
    print(f"Precision   : {summary['precision_mean']:.6f} ± {summary['precision_std']:.6f}")
    print(f"Recall      : {summary['recall_mean']:.6f} ± {summary['recall_std']:.6f}")
    print(f"Sensitivity : {summary['sensitivity_mean']:.6f} ± {summary['sensitivity_std']:.6f}")
    print(f"Specificity : {summary['specificity_mean']:.6f} ± {summary['specificity_std']:.6f}")
    print(f"F1-score    : {summary['f1_score_mean']:.6f} ± {summary['f1_score_std']:.6f}")
    print(f"MAE         : {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")

    print("\nConfusion Matrix Totals:")
    print(f"TP : {results_df['tp'].sum()}")
    print(f"TN : {results_df['tn'].sum()}")
    print(f"FP : {results_df['fp'].sum()}")
    print(f"FN : {results_df['fn'].sum()}")

    print("\nSaved:")
    print(f"Per-image metrics : {per_image_csv}")
    print(f"Summary metrics   : {summary_csv}")
    print(f"Visualizations    : {CFG.vis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    test_model()