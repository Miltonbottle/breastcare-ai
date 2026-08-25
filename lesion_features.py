"""Extract objective geometric features from a binary lesion segmentation mask.

The measurements in this module describe mask geometry only. They do not
provide a diagnosis or a benign/malignant classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def extract_lesion_features(mask: np.ndarray) -> dict[str, int | float | None]:
    """Extract geometric features from the largest foreground component.

    Args:
        mask: A two-dimensional binary NumPy array. Any value greater than
            zero is treated as foreground.

    Returns:
        A JSON-serializable dictionary containing the primary lesion's area,
        bounding box, aspect ratio, perimeter, circularity, centroid, and the
        count of foreground connected components. With no foreground pixels,
        all geometric measurements are zero and centroid coordinates are None.

    Raises:
        TypeError: If ``mask`` is not a NumPy array.
        ValueError: If ``mask`` is not two-dimensional or has no pixels.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array.")
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional binary array.")
    if mask.size == 0:
        raise ValueError("mask must contain at least one pixel.")

    binary_mask = (mask > 0).astype(np.uint8)
    total_pixels = binary_mask.size
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    connected_components = component_count - 1  # Exclude background label 0.

    if connected_components == 0:
        return {
            "lesion_area_pixels": 0,
            "lesion_area_percentage": 0.0,
            "bounding_box_x": 0,
            "bounding_box_y": 0,
            "bounding_box_width": 0,
            "bounding_box_height": 0,
            "aspect_ratio": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "centroid_x": None,
            "centroid_y": None,
            "connected_components": 0,
        }

    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    primary_label = int(np.argmax(foreground_areas)) + 1
    primary_stats = stats[primary_label]
    area = int(primary_stats[cv2.CC_STAT_AREA])
    x = int(primary_stats[cv2.CC_STAT_LEFT])
    y = int(primary_stats[cv2.CC_STAT_TOP])
    width = int(primary_stats[cv2.CC_STAT_WIDTH])
    height = int(primary_stats[cv2.CC_STAT_HEIGHT])

    primary_mask = (labels == primary_label).astype(np.uint8)
    contours, _ = cv2.findContours(
        primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

    return {
        "lesion_area_pixels": area,
        "lesion_area_percentage": float(area / total_pixels * 100.0),
        "bounding_box_x": x,
        "bounding_box_y": y,
        "bounding_box_width": width,
        "bounding_box_height": height,
        "aspect_ratio": float(width / height) if height > 0 else 0.0,
        "perimeter": perimeter,
        "circularity": circularity,
        "centroid_x": float(centroids[primary_label][0]),
        "centroid_y": float(centroids[primary_label][1]),
        "connected_components": int(connected_components),
    }


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a mask image as a two-dimensional binary NumPy array."""
    mask_path = Path(mask_path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Binary mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Invalid or unsupported mask image: {mask_path}")
    return (mask > 0).astype(np.uint8)


def main() -> int:
    """Run feature extraction for a binary mask provided on the command line."""
    parser = argparse.ArgumentParser(
        description="Extract objective geometric features from a binary mask."
    )
    parser.add_argument("mask_path", help="Path to a binary mask image.")
    args = parser.parse_args()

    try:
        features = extract_lesion_features(load_binary_mask(args.mask_path))
    except (FileNotFoundError, TypeError, ValueError, cv2.error) as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(features, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
