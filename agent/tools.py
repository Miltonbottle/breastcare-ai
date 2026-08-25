"""Tool functions used by the bounded BreastCareAgent workflow."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis_engine import analyze_lesion
from inference import get_final_segmentation_output, read_and_preprocess, save_outputs
from lesion_features import extract_lesion_features


def validate_image(image_path: str | Path) -> dict[str, Any]:
    """Validate and preprocess an image using the existing inference helper."""
    original_rgb, image_tensor = read_and_preprocess(Path(image_path))
    return {
        "original_rgb": original_rgb,
        "image_tensor": image_tensor,
        "width": int(original_rgb.shape[1]),
        "height": int(original_rgb.shape[0]),
        "pixel_std": float(np.std(original_rgb)),
    }


def decide_segmentation(validated_image: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the validated image meets minimum workflow conditions."""
    image_tensor = validated_image["image_tensor"]
    if tuple(image_tensor.shape) != (1, 3, 256, 256):
        return {
            "should_segment": False,
            "reason": "Preprocessed image does not match the model input contract.",
        }
    if min(validated_image["width"], validated_image["height"]) < 64:
        return {
            "should_segment": False,
            "reason": "Source image resolution is below the 64-pixel minimum workflow threshold.",
        }
    if validated_image["pixel_std"] <= 1.0:
        return {
            "should_segment": False,
            "reason": "Source image has insufficient pixel variation for segmentation review.",
        }
    return {
        "should_segment": True,
        "reason": "Image passed input-contract, resolution, and pixel-variation checks.",
    }


def run_segmentation(
    model: torch.nn.Module,
    device: torch.device,
    model_lock: threading.Lock,
    validated_image: dict[str, Any],
) -> dict[str, Any]:
    """Run the already-loaded model using the existing inference output logic."""
    image_tensor = validated_image["image_tensor"].to(device)
    with model_lock, torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        logits = get_final_segmentation_output(model(image_tensor))
        binary_mask = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start_time

    mask = binary_mask[0, 0].cpu().numpy()
    return {
        "mask": mask,
        "inference_seconds": inference_seconds,
    }


def check_usable_mask(mask: np.ndarray) -> dict[str, Any]:
    """Check whether segmentation produced usable two-dimensional foreground."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return {"usable": False, "reason": "Segmentation mask has an invalid shape."}

    foreground_pixels = int(np.count_nonzero(mask))
    if foreground_pixels == 0:
        return {
            "usable": False,
            "reason": "Segmentation produced an empty foreground mask.",
            "foreground_pixels": 0,
        }
    return {
        "usable": True,
        "foreground_pixels": foreground_pixels,
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
    }


def extract_features(mask: np.ndarray) -> dict[str, Any]:
    """Extract geometry using the existing lesion_features module."""
    return extract_lesion_features(mask)


def analyze_geometry(features: dict[str, Any]) -> dict[str, Any]:
    """Generate bounded geometry analysis using the existing analysis engine."""
    return analyze_lesion(features)


def quality_check(mask: np.ndarray, features: dict[str, Any]) -> dict[str, Any]:
    """Route mask geometry to accepted or review-required before reporting."""
    area_percentage = float(features["lesion_area_percentage"])
    components = int(features["connected_components"])
    flags: list[str] = []
    if area_percentage < 0.1:
        flags.append("Very small foreground area; review segmentation visibility.")
    if area_percentage >= 50.0:
        flags.append("Very large foreground area; review segmentation extent.")
    if components > 1:
        flags.append("Multiple foreground components; the largest component drives geometry.")

    outcome = "review_required" if flags else "accepted"
    return {
        "status": "review_required" if flags else "passed",
        "outcome": outcome,
        "reason": (
            "One or more segmentation sanity checks require review."
            if flags
            else "Mask passed configured segmentation sanity checks."
        ),
        "mask_dimensions": {"width": int(mask.shape[1]), "height": int(mask.shape[0])},
        "foreground_pixels": int(np.count_nonzero(mask)),
        "flags": flags,
    }


def save_segmentation_outputs(
    validated_image: dict[str, Any],
    mask: np.ndarray,
    image_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save artifacts only after the mask has passed usability checks."""
    return save_outputs(
        validated_image["original_rgb"], mask, Path(image_path), Path(output_dir)
    )


def generate_report(
    validated_image: dict[str, Any],
    segmentation: dict[str, Any],
    features: dict[str, Any],
    analysis: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the structured, non-diagnostic output of the agent workflow."""
    return {
        "image_width": validated_image["width"],
        "image_height": validated_image["height"],
        "output_paths": segmentation["output_paths"],
        "inference_seconds": segmentation["inference_seconds"],
        "features": features,
        "analysis": analysis,
        "segmentation_quality": quality,
        "workflow_outcome": quality["outcome"],
    }
