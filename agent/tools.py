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
    }


def decide_segmentation(validated_image: dict[str, Any]) -> dict[str, Any]:
    """Make the bounded decision to segment a successfully validated image."""
    image_tensor = validated_image["image_tensor"]
    should_segment = tuple(image_tensor.shape) == (1, 3, 256, 256)
    return {
        "should_segment": should_segment,
        "reason": "Image passed validation and matches the model input contract."
        if should_segment
        else "Image does not match the expected model input contract.",
    }


def run_segmentation(
    model: torch.nn.Module,
    device: torch.device,
    model_lock: threading.Lock,
    validated_image: dict[str, Any],
    image_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the already-loaded model and save output artifacts through inference.py."""
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
    output_paths = save_outputs(
        validated_image["original_rgb"], mask, Path(image_path), Path(output_dir)
    )
    return {
        "mask": mask,
        "output_paths": output_paths,
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
    """Perform transparent mask sanity checks without clinical interpretation."""
    area_percentage = float(features["lesion_area_percentage"])
    components = int(features["connected_components"])
    flags: list[str] = []
    if area_percentage < 0.1:
        flags.append("Very small foreground area; review segmentation visibility.")
    if area_percentage >= 50.0:
        flags.append("Very large foreground area; review segmentation extent.")
    if components > 1:
        flags.append("Multiple foreground components; the largest component drives geometry.")

    return {
        "status": "review_required" if flags else "passed",
        "mask_dimensions": {"width": int(mask.shape[1]), "height": int(mask.shape[0])},
        "foreground_pixels": int(np.count_nonzero(mask)),
        "flags": flags,
    }


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
    }
