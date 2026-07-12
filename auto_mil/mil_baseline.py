from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import dump_yaml
from .baseline_registry import assert_mil_baseline_root, assert_models_available
from .state import json_ready


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    stage: str
    model_name: str
    epochs: int
    lr: float
    dropout: float | None
    balanced_sampler: bool
    notes: str


@dataclass(frozen=True)
class RunResult:
    recipe: Recipe
    status: str
    command: list[str]
    config_path: Path
    stdout_path: Path
    log_dir: Path
    metrics: dict[str, Any]
    error: str | None = None

    @property
    def score(self) -> float:
        metric = self.metrics.get("test_macro_auc")
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            metric = self.metrics.get("val_macro_auc")
        try:
            return float(metric)
        except (TypeError, ValueError):
            return float("-inf")


def run_result_to_payload(result: RunResult) -> dict[str, Any]:
    return json_ready(asdict(result))


def run_result_from_payload(payload: dict[str, Any]) -> RunResult:
    recipe = Recipe(**payload["recipe"])
    return RunResult(
        recipe=recipe,
        status=str(payload["status"]),
        command=[str(x) for x in payload.get("command", [])],
        config_path=Path(payload["config_path"]),
        stdout_path=Path(payload["stdout_path"]),
        log_dir=Path(payload["log_dir"]),
        metrics=dict(payload.get("metrics", {})),
        error=payload.get("error"),
    )


def _deep_update_if_present(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: Any = root
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict) and parts[-1] in current:
        current[parts[-1]] = value


def build_mil_config(
    recipe: Recipe,
    base_config_path: Path,
    output_config_path: Path,
    dataset_name: str,
    dataset_csv_path: Path,
    log_root_dir: Path,
    num_classes: int,
    in_dim: int,
    device: int,
    num_workers: int,
    best_metric: str,
) -> Path:
    with base_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("General", {})
    cfg["General"]["MODEL_NAME"] = recipe.model_name
    cfg["General"]["num_classes"] = int(num_classes)
    cfg["General"]["num_epochs"] = int(recipe.epochs)
    cfg["General"]["device"] = int(device)
    cfg["General"]["num_workers"] = int(num_workers)
    cfg["General"]["best_model_metric"] = best_metric

    cfg.setdefault("Dataset", {})
    cfg["Dataset"]["DATASET_NAME"] = dataset_name
    cfg["Dataset"]["dataset_csv_path"] = str(dataset_csv_path)
    cfg["Dataset"].pop("dataset_root_dir", None)
    cfg["Dataset"].setdefault("balanced_sampler", {})
    if isinstance(cfg["Dataset"]["balanced_sampler"], dict):
        cfg["Dataset"]["balanced_sampler"]["use"] = bool(recipe.balanced_sampler)
        cfg["Dataset"]["balanced_sampler"].setdefault("replacement", True)

    cfg.setdefault("Logs", {})
    cfg["Logs"]["log_root_dir"] = str(log_root_dir)

    cfg.setdefault("Model", {})
    cfg["Model"]["in_dim"] = int(in_dim)
    if recipe.dropout is not None and "dropout" in cfg["Model"]:
        cfg["Model"]["dropout"] = float(recipe.dropout)

    _deep_update_if_present(cfg, "Model.optimizer.adam_config.lr", float(recipe.lr))
    _deep_update_if_present(cfg, "Model.optimizer.adamw_config.lr", float(recipe.lr))

    dump_yaml(cfg, output_config_path)
    return output_config_path


def _read_best_metrics(log_root_dir: Path) -> dict[str, Any]:
    candidates = sorted(log_root_dir.rglob("Best_Log*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(log_root_dir.rglob("Log_seed*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}

    path = candidates[-1]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"metrics_path": str(path)}
    row = rows[0]
    metrics: dict[str, Any] = {"metrics_path": str(path)}
    for key, value in row.items():
        if key in {"val_confusion_mat", "test_confusion_mat"}:
            metrics[key] = value
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = value
    return metrics


def run_recipe(
    recipe: Recipe,
    *,
    python: Path,
    mil_baseline_dir: Path,
    dataset_name: str,
    dataset_csv_path: Path,
    output_dir: Path,
    num_classes: int,
    in_dim: int,
    device: int,
    num_workers: int,
    best_metric: str,
    dry_run: bool = False,
    timeout_seconds: int | None = None,
) -> RunResult:
    mil_baseline_dir = assert_mil_baseline_root(mil_baseline_dir)
    assert_models_available([recipe.model_name], mil_baseline_dir)

    config_dir = output_dir / "configs"
    stdout_dir = output_dir / "stdout"
    log_root_dir = output_dir / "mil_logs" / recipe.recipe_id
    config_path = config_dir / f"{recipe.recipe_id}.yaml"
    stdout_path = stdout_dir / f"{recipe.recipe_id}.log"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    log_root_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = mil_baseline_dir / "configs" / f"{recipe.model_name}.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing MIL_BASELINE config: {base_config_path}")

    build_mil_config(
        recipe,
        base_config_path,
        config_path,
        dataset_name,
        dataset_csv_path,
        log_root_dir,
        num_classes,
        in_dim,
        device,
        num_workers,
        best_metric,
    )

    command = [str(python), "train_mil.py", "--yaml_path", str(config_path)]
    if dry_run:
        stdout_path.write_text(
            "DRY RUN\n" + json.dumps({"recipe": asdict(recipe), "command": command}, indent=2),
            encoding="utf-8",
        )
        return RunResult(
            recipe=recipe,
            status="dry_run",
            command=command,
            config_path=config_path,
            stdout_path=stdout_path,
            log_dir=log_root_dir,
            metrics={},
        )

    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.run(
            command,
            cwd=str(mil_baseline_dir),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    metrics = _read_best_metrics(log_root_dir)
    status = "completed" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"train_mil.py exited with code {proc.returncode}"
    return RunResult(
        recipe=recipe,
        status=status,
        command=command,
        config_path=config_path,
        stdout_path=stdout_path,
        log_dir=log_root_dir,
        metrics=metrics,
        error=error,
    )
