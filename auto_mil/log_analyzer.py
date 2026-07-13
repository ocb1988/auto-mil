from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LogDiagnosis:
    category: str
    severity: str
    summary: str
    recommended_action: str
    matched_pattern: str | None = None
    traceback_tail: str | None = None


PATTERNS: list[tuple[str, str, str, str, str]] = [
    (
        "missing_dependency",
        "actionable",
        r"ModuleNotFoundError: No module named '([^']+)'|ImportError: No module named ([^\s]+)",
        "Missing Python dependency.",
        "Install the dependency in the active environment, or add a local fallback if the method can run without it.",
    ),
    (
        "cuda_oom",
        "actionable",
        r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|out of memory",
        "GPU memory was exhausted.",
        "Reduce patch count, hidden dimension, batch-equivalent work, workers, or switch to a smaller method/GPU.",
    ),
    (
        "cuda_runtime",
        "actionable",
        r"CUDA error|device-side assert|CUDNN_STATUS|no kernel image is available|invalid device ordinal",
        "CUDA runtime error.",
        "Check device id, CUDA/PyTorch compatibility, labels, tensor shapes, and rerun with CUDA_LAUNCH_BLOCKING=1 if needed.",
    ),
    (
        "metric_missing_class",
        "nonfatal_data",
        r"Number of classes in y_true not equal to the number of columns|Only one class is present in y_true|ROC AUC score is not defined",
        "Metric computation failed because a validation/test split lacks one or more classes.",
        "Use stratified patient-level splits with enough cases per class, report affected metrics as nan, or use repeated CV.",
    ),
    (
        "file_not_found",
        "actionable",
        r"FileNotFoundError|No such file or directory|The system cannot find the file specified",
        "A required file or path was missing.",
        "Check dataset paths, generated CSV paths, feature bag paths, and config file paths.",
    ),
    (
        "h5_key_error",
        "actionable",
        r"KeyError:.*features|object 'features' doesn't exist|Unable to open object",
        "H5 feature file is missing the expected key.",
        "Inspect representative H5 files and update the dataset loader or preprocessing key mapping.",
    ),
    (
        "shape_mismatch",
        "actionable",
        r"mat1 and mat2 shapes cannot be multiplied|size mismatch|shape mismatch|Expected .* got",
        "Tensor shape mismatch.",
        "Check feature dimension, model in_dim, number of classes, and model-specific input expectations.",
    ),
    (
        "nan_loss",
        "actionable",
        r"loss.*nan|nan.*loss|ValueError: Input contains NaN|contains infinity",
        "NaN or infinite values appeared during training/evaluation.",
        "Lower learning rate, check feature values, add gradient clipping, or inspect unstable loss terms.",
    ),
    (
        "timeout",
        "runtime",
        r"timed out|TimeoutExpired|TIMEOUT:",
        "Experiment exceeded the allowed runtime.",
        "Reduce budget or raise the timeout for this method.",
    ),
    (
        "keyboard_interrupt",
        "interrupted",
        r"KeyboardInterrupt",
        "Run was interrupted by the user or host.",
        "Resume from checkpoint when ready.",
    ),
]


def _read_text(path: Path, max_chars: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _traceback_tail(text: str, max_lines: int = 24) -> str | None:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            start = idx
    if start is None:
        error_lines = [line for line in lines if "Error" in line or "Exception" in line or "failed" in line.lower()]
        return "\n".join(error_lines[-max_lines:]) if error_lines else None
    return "\n".join(lines[start:][-max_lines:])


def diagnose_log_text(text: str) -> LogDiagnosis:
    if not text.strip():
        return LogDiagnosis(
            category="missing_log",
            severity="unknown",
            summary="No log text was available for diagnosis.",
            recommended_action="Check whether the command started and whether stdout/stderr paths were written.",
        )

    if re.search(r"^\s*DRY RUN\s*$", text, flags=re.IGNORECASE | re.MULTILINE):
        return LogDiagnosis(
            category="dry_run",
            severity="info",
            summary="This log is from a dry run and does not contain training execution.",
            recommended_action="Run without --dry-run to execute the experiment.",
        )

    for category, severity, pattern, summary, action in PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            detail = match.group(0)
            if category == "missing_dependency":
                missing = next((g for g in match.groups() if g), None)
                if missing:
                    summary = f"Missing Python dependency: {missing}."
            return LogDiagnosis(
                category=category,
                severity=severity,
                summary=summary,
                recommended_action=action,
                matched_pattern=detail[:500],
                traceback_tail=_traceback_tail(text),
            )

    if "Traceback (most recent call last):" in text:
        return LogDiagnosis(
            category="python_exception",
            severity="actionable",
            summary="Python traceback found, but it did not match a known failure class.",
            recommended_action="Inspect the traceback tail and add a classifier pattern if this recurs.",
            traceback_tail=_traceback_tail(text),
        )

    if re.search(r"Best Log CSV Saved|Global Log CSV Saved|Last Model Saved|K-Fold:.*Done", text, flags=re.IGNORECASE):
        return LogDiagnosis(
            category="no_failure_signature",
            severity="info",
            summary="No failure signature was found and the log contains completion markers.",
            recommended_action="No action needed unless metrics are missing or suspicious.",
        )

    return LogDiagnosis(
        category="unknown",
        severity="unknown",
        summary="No known failure signature was found.",
        recommended_action="Inspect the raw log manually and add a classifier pattern if useful.",
        traceback_tail=_traceback_tail(text),
    )


def diagnose_log_file(path: str | Path) -> LogDiagnosis:
    return diagnose_log_text(_read_text(Path(path)))


def diagnosis_to_payload(diagnosis: LogDiagnosis | None) -> dict[str, Any] | None:
    return asdict(diagnosis) if diagnosis is not None else None


def diagnosis_from_payload(payload: dict[str, Any] | None) -> LogDiagnosis | None:
    if not payload:
        return None
    return LogDiagnosis(
        category=str(payload.get("category", "unknown")),
        severity=str(payload.get("severity", "unknown")),
        summary=str(payload.get("summary", "")),
        recommended_action=str(payload.get("recommended_action", "")),
        matched_pattern=payload.get("matched_pattern"),
        traceback_tail=payload.get("traceback_tail"),
    )


def write_diagnosis_json(diagnosis: LogDiagnosis, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(diagnosis), indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path
