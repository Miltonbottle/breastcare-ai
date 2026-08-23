# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:06:15 2026

@author: USER
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated"
)


import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
#import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold

from hybridsegnet import HybridViTGABVSSMUNet
from losses import BreastLesionSegmentationLoss
#from losses import BCEDiceLoss

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)
# =========================================================
# Configuration
# =========================================================

class CFG:
    image_dir = r"C:\Users\USER\Downloads\Dataset_Final_1595\malignant\images"
    mask_dir = r"C:\Users\USER\Downloads\Dataset_Final_1595\malignant\masks"

    save_dir = "checkpoints"
    log_dir = "logs"
    
    resume_training = False
    resume_checkpoint = "checkpoints/fold_1_current_model.pth"

    image_size = 256

    num_folds = 2
    epochs = 150
    batch_size = 4

    lr = 1e-4
    weight_decay = 1e-4

    num_workers = 4
    seed = 42

    patience = 150

    device = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(CFG.save_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)

device = torch.device(CFG.device)
# =========================================================
# Reproducibility
# =========================================================
def load_checkpoint_for_resume(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cuda"
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_dice = checkpoint.get("best_val_dice", 0.0)

    print("=" * 80)
    print("RESUMING TRAINING")
    print("=" * 80)
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Resume Epoch    : {start_epoch}")
    print(f"Best Val Dice   : {best_val_dice:.6f}")
    print("=" * 80)

    return model, optimizer, scheduler, start_epoch, best_val_dice

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)


# =========================================================
# Dataset
# =========================================================

class BreastUltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()

        return image, mask


# =========================================================
# Augmentations
# =========================================================

def get_train_transform():
    return A.Compose([
        A.Resize(CFG.image_size, CFG.image_size),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),

        A.ElasticTransform(
            alpha=20,
            sigma=5,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        A.GaussNoise(
            var_limit=(5.0, 30.0),
            p=0.3
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


def get_valid_transform():
    return A.Compose([
        A.Resize(CFG.image_size, CFG.image_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


# =========================================================
# Metrics
# =========================================================

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        preds.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()

def deep_supervision_loss(outputs, masks, criterion):
    final_out, aux3, aux2, aux1 = outputs

    loss_final = criterion(final_out, masks)
    loss_aux3 = criterion(aux3, masks)
    loss_aux2 = criterion(aux2, masks)
    loss_aux1 = criterion(aux1, masks)

    total_loss = (
        0.5 * loss_final +
        0.2 * loss_aux1 +
        0.2 * loss_aux2 +
        0.1 * loss_aux3
    )

    return total_loss
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

# =========================================================
# Train One Epoch
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        
        if isinstance(outputs, tuple):
            loss = deep_supervision_loss(outputs, masks, criterion)
            logits = outputs[0]
        else:
            logits = outputs
            loss = criterion(logits, masks)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_dice = dice_score(logits.detach(), masks)
        batch_iou = iou_score(logits.detach(), masks)

        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "iou": f"{batch_iou:.4f}"
        })

    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)

    return epoch_loss, epoch_dice, epoch_iou


# =========================================================
# Validate One Epoch
# =========================================================

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = criterion(logits, masks)

        batch_dice = dice_score(logits, masks)
        batch_iou = iou_score(logits, masks)

        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou

        progress_bar.set_postfix({
            "val_loss": f"{loss.item():.4f}",
            "val_dice": f"{batch_dice:.4f}",
            "val_iou": f"{batch_iou:.4f}"
        })

    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)

    return epoch_loss, epoch_dice, epoch_iou


# =========================================================
# Save Checkpoint
# =========================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    fold,
    best_val_dice,
    train_loss,
    val_loss
):
    checkpoint = {
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_dice": best_val_dice,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    torch.save(checkpoint, path)


# =========================================================
# Get Image and Mask Paths
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
# Main 5-Fold Training
# =========================================================

def run_training():
    image_paths, mask_paths = get_data_paths(CFG.image_dir, CFG.mask_dir)

    print("=" * 80)
    print(f"Total Samples Found : {len(image_paths)}")
    print(f"Device              : {CFG.device}")
    print("=" * 80)

    kfold = KFold(
        n_splits=CFG.num_folds,
        shuffle=True,
        random_state=CFG.seed
    )

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        print("\n" + "=" * 80)
        print(f"FOLD {fold + 1}/{CFG.num_folds}")
        print("=" * 80)

        train_images = [image_paths[i] for i in train_idx]
        train_masks = [mask_paths[i] for i in train_idx]

        val_images = [image_paths[i] for i in val_idx]
        val_masks = [mask_paths[i] for i in val_idx]

        print(f"Training Samples   : {len(train_images)}")
        print(f"Validation Samples : {len(val_images)}")

        train_dataset = BreastUltrasoundDataset(
            train_images,
            train_masks,
            transform=get_train_transform()
        )

        val_dataset = BreastUltrasoundDataset(
            val_images,
            val_masks,
            transform=get_valid_transform()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True
        )

        model = build_model().to(CFG.device)

        criterion = BreastLesionSegmentationLoss(
            bce_weight=0.3,
            dice_weight=0.3,
            focal_tversky_weight=0.15,
            boundary_weight=0.25,
            alpha=0.3,
            beta=0.7,
            gamma=0.75
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8
        )

        best_val_dice = 0.0
        best_epoch = 0
        patience_counter = 0
        start_epoch = 1
        best_val_dice = 0.0
        history = []
        
        if CFG.resume_training and os.path.exists(CFG.resume_checkpoint):
            model, optimizer, scheduler, start_epoch, best_val_dice = load_checkpoint_for_resume(
                checkpoint_path=CFG.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
        for epoch in range(start_epoch, CFG.epochs + 1):
            print("\n" + "-" * 80)
            print(f"Fold {fold + 1} | Epoch {epoch}/{CFG.epochs}")
            print("-" * 80)
        
            train_loss, train_dice, train_iou = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )
        
            val_loss, val_dice, val_iou = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device
            )
        
            scheduler.step(val_dice)
        
            current_lr = optimizer.param_groups[0]["lr"]
        
            print(f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.6f} | Train IoU: {train_iou:.6f}")
            print(f"Val Loss  : {val_loss:.6f} | Val Dice  : {val_dice:.6f} | Val IoU  : {val_iou:.6f}")
            print(f"Learning Rate: {current_lr:.8f}")
        
            current_model_path = os.path.join(
                CFG.save_dir,
                f"fold_{fold + 1}_current_model.pth"
            )
        
            save_checkpoint(
                path=current_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                fold=fold + 1,
                best_val_dice=best_val_dice,
                train_loss=train_loss,
                val_loss=val_loss
            )
        
            if val_dice > best_val_dice:
                best_val_dice = val_dice
        
                best_model_path = os.path.join(
                    CFG.save_dir,
                    f"fold_{fold + 1}_best_model.pth"
                )
        
                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    fold=fold + 1,
                    best_val_dice=best_val_dice,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
        
                print(f"Best model saved at epoch {epoch}")
                print(f"Best Validation Dice: {best_val_dice:.6f}")

            else:
                patience_counter += 1

                print(
                    f"No improvement. Early stopping counter: "
                    f"{patience_counter}/{CFG.patience}"
                )

            log_path = os.path.join(
                CFG.log_dir,
                f"fold_{fold + 1}_training_log.csv"
            )

            pd.DataFrame(history).to_csv(log_path, index=False)

            if patience_counter >= CFG.patience:
                print("\nEarly stopping triggered.")
                break

        fold_result = {
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "best_model_path": os.path.join(
                CFG.save_dir,
                f"fold_{fold + 1}_best_model.pth"
            )
        }

        all_fold_results.append(fold_result)

        print("\n" + "=" * 80)
        print(f"Fold {fold + 1} Finished")
        print(f"Best Epoch      : {best_epoch}")
        print(f"Best Val Dice   : {best_val_dice:.6f}")
        print("=" * 80)

    final_results_path = os.path.join(
        CFG.log_dir,
        "cross_validation_results.csv"
    )

    pd.DataFrame(all_fold_results).to_csv(final_results_path, index=False)

    mean_dice = np.mean([x["best_val_dice"] for x in all_fold_results])
    std_dice = np.std([x["best_val_dice"] for x in all_fold_results])

    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Mean Validation Dice : {mean_dice:.6f}")
    print(f"Std Validation Dice  : {std_dice:.6f}")
    print(f"Results saved to     : {final_results_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_training()