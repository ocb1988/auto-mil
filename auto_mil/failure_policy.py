from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .log_analyzer import LogDiagnosis
from .mil_baseline import Recipe
from .state import json_ready


@dataclass(frozen=True)
class FailureAction:
    category: str
    action: str
    retryable: bool
    summary: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def action_to_payload(action: FailureAction) -> dict[str, Any]:
    return json_ready(asdict(action))


def _memory_saving_overrides() -> dict[str, Any]:
    return {
        "General.num_workers": 0,
        "Model.hidden_dim": 256,
        "Model.D": 64,
        "Model.L": 256,
        "Model.aggregate_num": 64,
        "Model.k_neighbors": 4,
        "Model.k_components": 5,
        "Model.dropout": 0.25,
    }


def _nan_safety_overrides() -> dict[str, Any]:
    return {
        "General.num_workers": 0,
        "Model.dropout": 0.25,
        "Model.scheduler.warmup": 1,
    }


def decide_failure_action(diagnosis: LogDiagnosis | None) -> FailureAction:
    if diagnosis is None:
        return FailureAction(
            category="none",
            action="no_action",
            retryable=False,
            summary="No failure diagnosis is available.",
        )

    category = diagnosis.category
    if category == "missing_dependency":
        return FailureAction(
            category=category,
            action="ask_user",
            retryable=False,
            summary="A dependency is missing; pause for an explicit install or environment fix.",
            notes="Auto-MIL does not install packages automatically during an experiment run.",
        )
    if category == "cuda_oom":
        return FailureAction(
            category=category,
            action="retry_with_overrides",
            retryable=True,
            summary="Retry once with lower memory-pressure settings.",
            config_overrides=_memory_saving_overrides(),
            notes="Only keys already present in the target MIL_BASELINE YAML are changed.",
        )
    if category == "nan_loss":
        return FailureAction(
            category=category,
            action="retry_with_overrides",
            retryable=True,
            summary="Retry once with a lower learning rate and more conservative regularization.",
            config_overrides=_nan_safety_overrides(),
            notes="Learning rate is reduced in the retry recipe.",
        )
    if category == "timeout":
        return FailureAction(
            category=category,
            action="retry_with_overrides",
            retryable=True,
            summary="Retry once with a shorter budget to preserve the search loop.",
            notes="Epoch count is reduced in the retry recipe.",
        )
    if category == "keyboard_interrupt":
        return FailureAction(
            category=category,
            action="resume",
            retryable=True,
            summary="Resume or rerun the interrupted recipe from checkpoint policy.",
        )
    if category == "metric_missing_class":
        return FailureAction(
            category=category,
            action="ask_user",
            retryable=False,
            summary="The split or fold appears to miss a class; pause before changing split policy.",
        )
    if category in {"file_not_found", "h5_key_error", "shape_mismatch"}:
        return FailureAction(
            category=category,
            action="ask_user",
            retryable=False,
            summary="The data/config contract needs inspection before another training attempt.",
        )
    if category == "cuda_runtime":
        return FailureAction(
            category=category,
            action="inspect_log",
            retryable=False,
            summary="CUDA failed outside a clear OOM pattern; inspect driver/device/runtime details.",
        )
    if category in {"dry_run", "no_failure_signature"}:
        return FailureAction(
            category=category,
            action="no_action",
            retryable=False,
            summary="No corrective retry is needed.",
        )
    return FailureAction(
        category=category,
        action="inspect_log",
        retryable=False,
        summary="No safe automatic retry policy exists for this failure category.",
    )


def make_retry_recipe(recipe: Recipe, action: FailureAction, retry_index: int = 1) -> Recipe | None:
    if not action.retryable:
        return None

    lr = recipe.lr
    epochs = recipe.epochs
    overrides = dict(recipe.config_overrides)
    overrides.update(action.config_overrides)
    if action.category == "nan_loss":
        lr = max(float(recipe.lr) * 0.5, 1e-7)
    if action.category == "timeout":
        epochs = max(1, int(recipe.epochs) // 2)

    return Recipe(
        recipe_id=f"{recipe.recipe_id}_retry{retry_index}_{action.category}",
        stage=f"{recipe.stage}_retry",
        model_name=recipe.model_name,
        epochs=epochs,
        lr=lr,
        dropout=recipe.dropout,
        balanced_sampler=recipe.balanced_sampler,
        notes=f"Retry of {recipe.recipe_id}: {action.summary}",
        config_overrides=overrides,
    )
