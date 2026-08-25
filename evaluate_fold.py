"""Evaluate the original Fold 1 validation subset for the trained segmenter.

This script reproduces the validation partition defined in train.py. Results
are validation results, not independent test-set performance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

from hybridsegnet import HybridViTGABVSSMUNet


IMAGE_SIZE = 256
THRESHOLD = 0.5
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = Path(
    r"C:\Users\ASUS\Downloads\Dataset_B1\Dataset_B\Dataset_Final_1595"
    r"\Dataset_Final_1595\malignant\images"
)
DEFAULT_MASK_DIR = Path(
    r"C:\Users\ASUS\Downloads\Dataset_B1\Dataset_B\Dataset_Final_1595"
    r"\Dataset_Final_1595\malignant\masks"
)
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "fold_1_best_model.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_model() -> HybridViTGABVSSMUNet:
    """Build the exact HybridViTGABVSSMUNet configuration from train.py."""
    return HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(1, 1, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1,
    )


def get_data_paths(image_dir: Path, mask_dir: Path) -> tuple[list[Path], list[Path]]:
    """Reproduce train.py's sorted, same-name mask-paired path construction."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_files = sorted(
        file_name
        for file_name in os.listdir(image_dir)
        if file_name.lower().endswith(VALID_EXTENSIONS)
    )
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    for file_name in image_files:
        image_path = image_dir / file_name
        mask_path = mask_dir / file_name
        if mask_path.exists():
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f"Warning: mask not found for {file_name}")

    if not image_paths:
        raise ValueError("No image/mask pairs were found.")
    return image_paths, mask_paths


def get_fold_one_validation_paths(
    image_paths: list[Path], mask_paths: list[Path]
) -> tuple[list[Path], list[Path]]:
    """Return val_idx from the first split yielded by train.py's KFold setup."""
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    _, validation_indices = next(kfold.split(image_paths))
    return (
        [image_paths[index] for index in validation_indices],
        [mask_paths[index] for index in validation_indices],
    )


def get_validation_transform() -> A.Compose:
    """Return the exact validation transform from train.py."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def load_model(checkpoint_path: Path, device: torch.device) -> HybridViTGABVSSMUNet:
    """Load the saved model_state_dict from the Fold 1 checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        if not isinstance(state_dict, dict):
            raise TypeError("The checkpoint does not contain a valid model state dictionary.")

        model = build_model()
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"Could not load model from '{checkpoint_path}': {exc}") from exc


def get_final_segmentation_output(output: Any) -> torch.Tensor:
    """Use final logits when a tensor or deep-supervision output is returned."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty deep-supervision output.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(output).__name__}")
    return output


def calculate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate per-image binary segmentation metrics."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(prediction, target).sum()
    false_positive = np.logical_and(prediction, ~target).sum()
    false_negative = np.logical_and(~prediction, target).sum()
    smooth = 1e-6

    return {
        "dice": float((2 * true_positive + smooth) / (2 * true_positive + false_positive + false_negative + smooth)),
        "iou": float((true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)),
        "precision": float((true_positive + smooth) / (true_positive + false_positive + smooth)),
        "recall": float((true_positive + smooth) / (true_positive + false_negative + smooth)),
    }


def load_image_and_mask(
    image_path: Path, mask_path: Path, transform: A.Compose
) -> tuple[torch.Tensor, np.ndarray]:
    """Load an RGB image and binary mask using the validation transform."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Invalid or unsupported image: {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Invalid or unsupported ground-truth mask: {mask_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = (mask > 127).astype(np.float32)
    transformed = transform(image=image_rgb, mask=mask)
    image_tensor = transformed["image"]
    mask_np = (transformed["mask"].numpy() > 0.5).astype(np.uint8)
    return image_tensor, mask_np


def evaluate_fold_one(
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    mask_dir: str | Path = DEFAULT_MASK_DIR,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, float | int | Path | str]]:
    """Evaluate only the exact Fold 1 validation partition from train.py."""
    image_dir = Path(image_dir).expanduser().resolve()
    mask_dir = Path(mask_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, mask_paths = get_data_paths(image_dir, mask_dir)
    validation_images, validation_masks = get_fold_one_validation_paths(image_paths, mask_paths)
    model = load_model(checkpoint_path, device)
    transform = get_validation_transform()
    rows: list[dict[str, float | str]] = []

    for image_path, mask_path in tqdm(
        zip(validation_images, validation_masks),
        total=len(validation_images),
        desc="Fold 1 validation evaluation",
    ):
        image_tensor, target = load_image_and_mask(image_path, mask_path, transform)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = get_final_segmentation_output(model(image_tensor))
            prediction = (torch.sigmoid(logits) > THRESHOLD).to(torch.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        metrics = calculate_metrics(prediction[0, 0].cpu().numpy(), target)
        rows.append({"image": image_path.name, **metrics, "inference_time": inference_time})

    results = pd.DataFrame(
        rows, columns=["image", "dice", "iou", "precision", "recall", "inference_time"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "fold_1_validation_metrics.csv"
    results.to_csv(metrics_path, index=False)

    summary: dict[str, float | int | Path | str] = {
        "validation_images": len(results),
        "mean_dice": float(results["dice"].mean()),
        "std_dice": float(results["dice"].std()),
        "median_dice": float(results["dice"].median()),
        "mean_iou": float(results["iou"].mean()),
        "std_iou": float(results["iou"].std()),
        "mean_precision": float(results["precision"].mean()),
        "mean_recall": float(results["recall"].mean()),
        "mean_inference_time": float(results["inference_time"].mean()),
        "median_inference_time": float(results["inference_time"].median()),
        "device": str(device),
        "metrics_path": metrics_path,
    }
    summary_path = output_dir / "fold_1_validation_summary.txt"
    summary["summary_path"] = summary_path
    summary_path.write_text(format_report(summary), encoding="utf-8")
    return results, summary


def format_report(summary: dict[str, float | int | Path | str]) -> str:
    """Format a clear Fold 1 validation-only report."""
    return "\n".join([
        "Fold 1 validation evaluation",
        "=" * 40,
        f"VALIDATION IMAGES: {summary['validation_images']}",
        f"DEVICE: {summary['device']}",
        f"MEAN DICE: {summary['mean_dice']:.6f}",
        f"STD DICE: {summary['std_dice']:.6f}",
        f"MEDIAN DICE: {summary['median_dice']:.6f}",
        f"MEAN IOU: {summary['mean_iou']:.6f}",
        f"STD IOU: {summary['std_iou']:.6f}",
        f"MEAN PRECISION: {summary['mean_precision']:.6f}",
        f"MEAN RECALL: {summary['mean_recall']:.6f}",
        f"MEAN INFERENCE TIME: {summary['mean_inference_time']:.3f} seconds",
        f"MEDIAN INFERENCE TIME: {summary['median_inference_time']:.3f} seconds",
        f"PER-IMAGE METRICS: {summary['metrics_path']}",
        f"SUMMARY FILE: {summary['summary_path']}",
        "",
        "This is Fold 1 validation evaluation, not a test-set result.",
        "It does not represent independent test performance.",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the exact Fold 1 validation subset.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Malignant image directory.")
    parser.add_argument("--mask-dir", default=str(DEFAULT_MASK_DIR), help="Matching mask directory.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Fold 1 checkpoint path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Results output directory.")
    args = parser.parse_args()

    try:
        _, summary = evaluate_fold_one(
            args.image_dir, args.mask_dir, args.checkpoint, args.output_dir
        )
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, OSError) as exc:
        print(f"Fold 1 validation evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
