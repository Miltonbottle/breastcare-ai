"""Generate a neutral, review-oriented analysis from segmentation geometry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from numbers import Real
from pathlib import Path
from typing import Any

from lesion_features import extract_lesion_features, load_binary_mask


def _require_finite_number(features: dict[str, Any], key: str) -> float:
    """Read a required finite numeric feature with a clear validation error."""
    if key not in features:
        raise ValueError(f"Missing required feature: {key}")
    value = features[key]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Feature '{key}' must be a finite number.")
    return float(value)


def _describe_extent(area_percentage: float) -> str:
    """Describe relative segmented area with explicit, non-clinical thresholds."""
    if area_percentage < 1.0:
        return "very small segmented area (<1% of image)"
    if area_percentage < 10.0:
        return "small segmented area (1% to <10% of image)"
    if area_percentage < 25.0:
        return "moderate segmented area (10% to <25% of image)"
    return "large segmented area (>=25% of image)"


def _describe_shape(aspect_ratio: float, circularity: float, area_percentage: float) -> str:
    """Return a neutral description based only on supplied geometric measures."""
    if area_percentage == 0:
        return "no foreground segmented region"

    elongation = (
        "approximately balanced in width and height"
        if 0.75 <= aspect_ratio <= 1.33
        else "elongated in its bounding-box dimensions"
    )
    boundary = (
        "with compact geometry"
        if circularity >= 0.75
        else "with intermediate geometric compactness"
        if circularity >= 0.50
        else "with irregular geometry"
    )
    return f"{elongation}, {boundary}"


def analyze_lesion(features: dict[str, Any]) -> dict[str, Any]:
    """Create a structured, non-diagnostic review summary from mask features.

    The input should be the dictionary returned by
    ``lesion_features.extract_lesion_features``. All conclusions are limited
    to the supplied segmentation geometry.
    """
    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary.")

    area_percentage = _require_finite_number(features, "lesion_area_percentage")
    aspect_ratio = _require_finite_number(features, "aspect_ratio")
    circularity = _require_finite_number(features, "circularity")
    connected_components_value = _require_finite_number(features, "connected_components")
    if area_percentage < 0 or aspect_ratio < 0 or circularity < 0:
        raise ValueError("Area percentage, aspect ratio, and circularity cannot be negative.")
    if connected_components_value < 0 or not connected_components_value.is_integer():
        raise ValueError("connected_components must be a non-negative integer.")
    connected_components = int(connected_components_value)

    review_flags: list[str] = []
    if connected_components == 0:
        review_flags.append("No foreground segmented region was identified; review the input mask.")
    if connected_components > 1:
        review_flags.append("Multiple segmented regions detected; review component selection.")
    if 0 < area_percentage < 0.1:
        review_flags.append("Extremely small segmented area (<0.1% of image); review segmentation visibility.")
    if area_percentage >= 50.0:
        review_flags.append("Extremely large segmented area (>=50% of image); review segmentation extent.")
    if circularity < 0.5 and area_percentage > 0:
        review_flags.append("Low circularity indicates irregular geometry; review the segmentation boundary.")
    if not review_flags:
        review_flags.append("No geometry-based review flags were triggered by the configured thresholds.")

    return {
        "lesion_area_percentage": area_percentage,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "connected_components": connected_components,
        "lesion_extent": _describe_extent(area_percentage),
        "shape_description": _describe_shape(aspect_ratio, circularity, area_percentage),
        "review_flags": review_flags,
        "limitations": [
            "This summary uses only pixels in the predicted segmentation mask.",
            "Image quality, acquisition context, and non-geometric information are not evaluated.",
            "Segmentation errors can affect every reported measurement.",
        ],
        "disclaimer": (
            "Geometric measurements derived from segmentation are not a medical diagnosis "
            "and require clinician review in the appropriate context."
        ),
    }


def main() -> int:
    """Load one binary mask and print its feature-based analysis as JSON."""
    parser = argparse.ArgumentParser(
        description="Generate a neutral geometric analysis from a binary mask."
    )
    parser.add_argument("mask_path", help="Path to a binary mask image.")
    args = parser.parse_args()

    try:
        features = extract_lesion_features(load_binary_mask(Path(args.mask_path)))
        analysis = analyze_lesion(features)
    except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
