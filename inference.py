"""Run lesion segmentation on one ultrasound image.

This module produces segmentation artefacts only; it does not provide a
diagnosis or a malignant/benign classification.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from hybridsegnet import HybridViTGABVSSMUNet


IMAGE_SIZE = 256
THRESHOLD = 0.5
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "fold_1_best_model.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inference_results"


def build_model() -> HybridViTGABVSSMUNet:
    """Build the architecture with the exact configuration from train.py."""
    return HybridViTGABVSSMUNet(
        in_channels=3,
        num_classes=1,
        encoder_dims=(32, 64, 128, 256),
        encoder_depths=(1, 1, 2, 2),
        encoder_heads=(1, 2, 4, 8),
        dropout=0.1,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> HybridViTGABVSSMUNet:
    """Load either the project's checkpoint dictionary or a raw state dict."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError("The checkpoint does not contain a valid model state dictionary.")

        model = build_model()
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"Could not load model from '{checkpoint_path}': {exc}") from exc


def read_and_preprocess(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Read an RGB image and apply the validation preprocessing exactly."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(
            f"Invalid or unsupported image: {image_path}. "
            "Use a readable PNG, JPG, BMP, TIFF, or similar image file."
        )

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(
        original_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    normalized = resized_rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)

    return original_rgb, tensor


def get_final_segmentation_output(output: Any) -> torch.Tensor:
    """Return final logits from either a tensor or deep-supervision output."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty deep-supervision output.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(output).__name__}")
    return output


def save_outputs(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Save original image, 256x256 binary mask, and original-size overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    original_path = output_dir / f"{stem}_original.png"
    mask_path = output_dir / f"{stem}_binary_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    mask_at_original_size = cv2.resize(
        mask_uint8,
        (original_rgb.shape[1], original_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    lesion_color = np.zeros_like(original_rgb)
    lesion_color[:, :, 0] = mask_at_original_size  # Red in RGB.
    overlay = cv2.addWeighted(original_rgb, 0.55, lesion_color, 0.45, 0)

    for path, image_rgb in (
        (original_path, original_rgb),
        (mask_path, mask_uint8),
        (overlay_path, overlay),
    ):
        image_to_write = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), image_to_write):
            raise OSError(f"Could not save output image: {path}")

    return {"original": original_path, "binary_mask": mask_path, "overlay": overlay_path}


def run_inference(
    image_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Segment one image and return paths plus non-diagnostic mask statistics."""
    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_rgb, input_tensor = read_and_preprocess(image_path)
    model = load_model(checkpoint_path, device)

    input_tensor = input_tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = get_final_segmentation_output(model(input_tensor))
        probabilities = torch.sigmoid(logits)
        binary_mask = (probabilities > THRESHOLD).to(torch.uint8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time

    mask_np = binary_mask[0, 0].cpu().numpy()
    output_paths = save_outputs(original_rgb, mask_np, image_path, output_dir)

    return {
        "input_image": image_path,
        "device": str(device),
        "inference_seconds": inference_seconds,
        "mask_size": tuple(mask_np.shape),
        "lesion_percentage": float(mask_np.mean() * 100.0),
        "output_paths": output_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BreastCare AI segmentation on one ultrasound image.")
    parser.add_argument("image_path", help="Path to one ultrasound image.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output images.")
    args = parser.parse_args()

    try:
        result = run_inference(args.image_path, args.checkpoint, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print("BreastCare AI Segmentation Complete")
    print(f"Input image                 : {result['input_image']}")
    print(f"Device                      : {result['device']}")
    print(f"Inference time              : {result['inference_seconds']:.3f} seconds")
    print(f"Predicted mask size         : {result['mask_size'][1]} x {result['mask_size'][0]}")
    print(f"Image predicted as lesion   : {result['lesion_percentage']:.2f}%")
    print(f"Original image              : {result['output_paths']['original']}")
    print(f"Binary mask                 : {result['output_paths']['binary_mask']}")
    print(f"Segmentation overlay        : {result['output_paths']['overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
