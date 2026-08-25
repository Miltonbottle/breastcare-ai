"""Evaluate BreastCare AI segmentation on one image with a ground-truth mask.

This utility reports segmentation agreement for one image only. It does not
represent overall model performance and does not provide a clinical diagnosis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from hybridsegnet import HybridViTGABVSSMUNet


IMAGE_SIZE = 256
THRESHOLD = 0.5
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = Path(
    r"C:\Users\ASUS\Downloads\Dataset_B1\Dataset_B\Dataset_Final_1595"
    r"\Dataset_Final_1595\malignant\images\000018.png"
)
DEFAULT_MASK_PATH = Path(
    r"C:\Users\ASUS\Downloads\Dataset_B1\Dataset_B\Dataset_Final_1595"
    r"\Dataset_Final_1595\malignant\masks\000018.png"
)
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "fold_1_best_model.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"


def build_model() -> HybridViTGABVSSMUNet:
    """Build the exact architecture configuration used in train.py."""
    return HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(1, 1, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> HybridViTGABVSSMUNet:
    """Load the project checkpoint dictionary or a raw model state dictionary."""
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


def load_image_and_preprocess(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Load RGB image and apply resize, ImageNet normalization, and ToTensor."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Invalid or unsupported input image: {image_path}")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(
        original_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    normalized = resized_rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return original_rgb, tensor


def load_ground_truth_mask(mask_path: Path) -> np.ndarray:
    """Load and nearest-neighbour resize a binary ground-truth mask to 256x256."""
    if not mask_path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Invalid or unsupported ground-truth mask: {mask_path}")

    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def get_final_segmentation_output(output: Any) -> torch.Tensor:
    """Use final logits if a model returns tensor or deep-supervision outputs."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty deep-supervision output.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(output).__name__}")
    return output


def calculate_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Calculate binary segmentation metrics for equally sized masks."""
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    true_positive = np.logical_and(prediction, ground_truth).sum()
    false_positive = np.logical_and(prediction, ~ground_truth).sum()
    false_negative = np.logical_and(~prediction, ground_truth).sum()
    smooth = 1e-6

    return {
        "dice": float((2 * true_positive + smooth) / (2 * true_positive + false_positive + false_negative + smooth)),
        "iou": float((true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)),
        "precision": float((true_positive + smooth) / (true_positive + false_positive + smooth)),
        "recall": float((true_positive + smooth) / (true_positive + false_negative + smooth)),
        "predicted_area_percentage": float(prediction.mean() * 100.0),
        "ground_truth_area_percentage": float(ground_truth.mean() * 100.0),
    }


def save_visualization(
    original_rgb: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    image_path: Path,
    output_dir: Path,
) -> Path:
    """Save original image, masks, and original-size red prediction overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_at_original_size = cv2.resize(
        prediction.astype(np.uint8),
        (original_rgb.shape[1], original_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    prediction_color = np.zeros_like(original_rgb)
    prediction_color[:, :, 0] = prediction_at_original_size * 255
    overlay = cv2.addWeighted(original_rgb, 0.55, prediction_color, 0.45, 0)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        (original_rgb, "Original Ultrasound", None),
        (ground_truth, "Ground-Truth Mask", "gray"),
        (prediction, "Predicted Mask", "gray"),
        (overlay, "Prediction Overlay", None),
    )
    for axis, (image, title, colour_map) in zip(axes, panels):
        axis.imshow(image, cmap=colour_map)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    visualization_path = output_dir / f"{image_path.stem}_evaluation.png"
    figure.savefig(visualization_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return visualization_path


def evaluate_single_image(
    image_path: str | Path,
    mask_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate one image/mask pair and return its metrics and output path."""
    image_path = Path(image_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_rgb, image_tensor = load_image_and_preprocess(image_path)
    ground_truth = load_ground_truth_mask(mask_path)
    model = load_model(checkpoint_path, device)

    image_tensor = image_tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = get_final_segmentation_output(model(image_tensor))
        prediction = (torch.sigmoid(logits) > THRESHOLD).to(torch.uint8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time

    prediction_np = prediction[0, 0].cpu().numpy()
    metrics = calculate_metrics(prediction_np, ground_truth)
    visualization_path = save_visualization(
        original_rgb, ground_truth, prediction_np, image_path, output_dir
    )

    return {
        "image_path": image_path,
        "device": str(device),
        "inference_seconds": inference_seconds,
        "visualization_path": visualization_path,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one ultrasound image and its ground-truth mask.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Ultrasound image path.")
    parser.add_argument("--mask", default=str(DEFAULT_MASK_PATH), help="Ground-truth mask path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Visualization output directory.")
    args = parser.parse_args()

    try:
        result = evaluate_single_image(args.image, args.mask, args.checkpoint, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("BreastCare AI Single-Image Segmentation Evaluation")
    print(f"IMAGE: {result['image_path'].name}")
    print(f"DEVICE: {result['device']}")
    print(f"DICE: {result['dice']:.6f}")
    print(f"IOU: {result['iou']:.6f}")
    print(f"PRECISION: {result['precision']:.6f}")
    print(f"RECALL: {result['recall']:.6f}")
    print(f"PREDICTED LESION AREA: {result['predicted_area_percentage']:.2f}%")
    print(f"GROUND TRUTH LESION AREA: {result['ground_truth_area_percentage']:.2f}%")
    print(f"INFERENCE TIME: {result['inference_seconds']:.3f} seconds")
    print(f"VISUALIZATION: {result['visualization_path']}")
    print("Note: this one-image result does not represent overall model performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
