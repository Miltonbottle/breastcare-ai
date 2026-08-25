"""Deterministic orchestration for the BreastCare AI analysis workflow."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import torch

from . import tools
from .schemas import AgentTraceStep, AgentWorkflowError


class BreastCareAgent:
    """Execute a bounded image-to-geometry workflow with an explicit audit trail."""

    name = "BreastCareAgent"

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        output_dir: str | Path,
        model_lock: threading.Lock | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_lock = model_lock or threading.Lock()

    @property
    def ready(self) -> bool:
        """Indicate whether the agent has a model instance ready for execution."""
        return self.model is not None

    def _run_step(
        self,
        trace: list[AgentTraceStep],
        tool_name: str,
        callback: Callable[[], Any],
        failure_message: str,
        status_code: int = 500,
    ) -> Any:
        """Execute one tool, recording completed or failed status in the trace."""
        try:
            value = callback()
            trace.append(AgentTraceStep(tool=tool_name, status="completed"))
            return value
        except Exception as exc:
            trace.append(
                AgentTraceStep(
                    tool=tool_name,
                    status="failed",
                    details={"reason": str(exc)},
                )
            )
            raise AgentWorkflowError(f"{failure_message}: {exc}", status_code, trace) from exc

    def execute(self, image_path: str | Path) -> dict[str, Any]:
        """Run validation, segmentation, geometry, quality checks, and reporting."""
        trace: list[AgentTraceStep] = []
        validated_image = self._run_step(
            trace,
            "validate_image",
            lambda: tools.validate_image(image_path),
            "Invalid image",
            status_code=400,
        )
        trace[-1].details = {
            "source_size": f"{validated_image['width']}x{validated_image['height']}",
            "reason": "Image decoded and preprocessed using the inference pipeline.",
        }
        decision = self._run_step(
            trace,
            "decide_segmentation",
            lambda: tools.decide_segmentation(validated_image),
            "Segmentation decision failed",
        )
        trace[-1].details = {
            "decision": "segment" if decision["should_segment"] else "stop",
            "reason": decision["reason"],
        }
        if not decision["should_segment"]:
            trace[-1].status = "stopped"
            raise AgentWorkflowError(decision["reason"], 422, trace)

        segmentation = self._run_step(
            trace,
            "run_segmentation",
            lambda: tools.run_segmentation(
                self.model,
                self.device,
                self.model_lock,
                validated_image,
            ),
            "Model inference failure",
        )
        trace[-1].details = {
            "mask_size": f"{segmentation['mask'].shape[1]}x{segmentation['mask'].shape[0]}",
            "inference_time_seconds": round(segmentation["inference_seconds"], 4),
        }
        usable_mask = self._run_step(
            trace,
            "check_usable_mask",
            lambda: tools.check_usable_mask(segmentation["mask"]),
            "Mask validation failure",
        )
        trace[-1].details = {
            "decision": "continue" if usable_mask["usable"] else "stop",
            "reason": usable_mask.get("reason", "Foreground mask is present."),
            "foreground_pixels": usable_mask.get("foreground_pixels", 0),
        }
        if not usable_mask["usable"]:
            trace[-1].status = "stopped"
            raise AgentWorkflowError(usable_mask["reason"], 422, trace)

        features = self._run_step(
            trace,
            "extract_features",
            lambda: tools.extract_features(segmentation["mask"]),
            "Feature extraction failure",
        )
        trace[-1].details = {
            "primary_area_percentage": round(features["lesion_area_percentage"], 4),
            "connected_components": features["connected_components"],
        }
        quality = self._run_step(
            trace,
            "quality_check",
            lambda: tools.quality_check(segmentation["mask"], features),
            "Segmentation quality check failure",
        )
        trace[-1].details = {
            "decision": quality["outcome"],
            "reason": quality["reason"],
            "flag_count": len(quality["flags"]),
        }
        analysis = self._run_step(
            trace,
            "analyze_geometry",
            lambda: tools.analyze_geometry(features),
            "Geometry analysis failure",
        )
        trace[-1].details = {
            "reason": "Geometry analysis generated from extracted mask features.",
        }
        output_paths = self._run_step(
            trace,
            "save_segmentation_outputs",
            lambda: tools.save_segmentation_outputs(
                validated_image, segmentation["mask"], image_path, self.output_dir
            ),
            "Output artifact failure",
        )
        segmentation["output_paths"] = output_paths
        trace[-1].details = {
            "reason": "Usable segmentation artifacts saved for review.",
        }
        report = self._run_step(
            trace,
            "generate_report",
            lambda: tools.generate_report(
                validated_image, segmentation, features, analysis, quality
            ),
            "Report generation failure",
        )
        trace[-1].details = {
            "outcome": quality["outcome"],
            "reason": "Structured report assembled after quality routing.",
        }
        report["agent_trace"] = {
            "agent": self.name,
            "steps": [step.to_dict() for step in trace],
        }
        return report
