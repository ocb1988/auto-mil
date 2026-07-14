from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .state import json_ready, now_iso


FOLD_RE = re.compile(r"(?:^|_)fold(?P<fold>\d+)(?:_|$)", re.IGNORECASE)
DEFAULT_METRICS = ["test_macro_auc", "test_bacc", "test_macro_f1", "test_acc", "val_macro_auc"]


@dataclass(frozen=True)
class CollectedRun:
    run_id: str
    experiment: str
    checkpoint: str
    stage: str
    status: str
    model_name: str
    fold: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_path: str | None = None
    config_path: str | None = None
    stdout_path: str | None = None
    log_dir: str | None = None
    error: str | None = None
    diagnosis_category: str | None = None
    diagnosis_summary: str | None = None


@dataclass(frozen=True)
class ModelResultSummary:
    experiment: str
    model_name: str
    status_scope: str
    n_completed: int
    n_total: int
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultBundle:
    generated_at: str
    root: str
    checkpoints: list[str]
    primary_metric: str
    metrics: list[str]
    runs: list[CollectedRun]
    summaries: list[ModelResultSummary]
    warnings: list[str] = field(default_factory=list)


def discover_checkpoints(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.name == "checkpoint.json":
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("checkpoint.json") if path.is_file())


def collect_results(
    root: str | Path,
    *,
    checkpoint_paths: list[str | Path] | None = None,
    primary_metric: str = "test_macro_auc",
    metrics: list[str] | None = None,
    include_failed: bool = True,
) -> ResultBundle:
    root = Path(root)
    checkpoints = [Path(path) for path in checkpoint_paths] if checkpoint_paths else discover_checkpoints(root)
    metric_names = metrics or DEFAULT_METRICS
    warnings: list[str] = []
    if not checkpoints:
        warnings.append(f"No checkpoint.json files found under {root}.")

    runs: list[CollectedRun] = []
    for checkpoint_path in checkpoints:
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except OSError as exc:
            warnings.append(f"Could not read {checkpoint_path}: {exc}")
            continue
        except json.JSONDecodeError as exc:
            warnings.append(f"Could not parse {checkpoint_path}: {exc}")
            continue
        runs.extend(
            _runs_from_checkpoint(
                root=root,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                metric_names=metric_names,
                include_failed=include_failed,
            )
        )
    summaries = summarize_runs(runs, metric_names)
    if runs and not any(primary_metric in run.metrics for run in runs):
        warnings.append(f"Primary metric {primary_metric!r} was not found in collected runs.")
    return ResultBundle(
        generated_at=now_iso(),
        root=str(root),
        checkpoints=[str(path) for path in checkpoints],
        primary_metric=primary_metric,
        metrics=metric_names,
        runs=runs,
        summaries=summaries,
        warnings=warnings,
    )


def summarize_runs(runs: list[CollectedRun], metric_names: list[str] | None = None) -> list[ModelResultSummary]:
    metric_names = metric_names or DEFAULT_METRICS
    groups: dict[tuple[str, str], list[CollectedRun]] = {}
    for run in runs:
        groups.setdefault((run.experiment, run.model_name), []).append(run)

    summaries: list[ModelResultSummary] = []
    for (experiment, model), rows in sorted(groups.items()):
        completed = [row for row in rows if row.status == "completed"]
        metric_summary: dict[str, dict[str, float | None]] = {}
        for metric in metric_names:
            values = [row.metrics[metric] for row in completed if metric in row.metrics]
            if values:
                metric_summary[metric] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            else:
                metric_summary[metric] = {"mean": None, "std": None, "min": None, "max": None}
        summaries.append(
            ModelResultSummary(
                experiment=experiment,
                model_name=model,
                status_scope="completed",
                n_completed=len(completed),
                n_total=len(rows),
                metrics=metric_summary,
            )
        )
    return summaries


def write_result_bundle(bundle: ResultBundle, output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results_index.json"
    runs_csv = output_dir / "runs.csv"
    summary_csv = output_dir / "model_summary.csv"
    report_md = output_dir / "manuscript_results.md"

    json_path.write_text(json.dumps(json_ready(asdict(bundle)), indent=2, ensure_ascii=True), encoding="utf-8")
    _write_runs_csv(bundle, runs_csv)
    _write_summary_csv(bundle, summary_csv)
    report_md.write_text(_render_markdown(bundle), encoding="utf-8")
    return json_path, runs_csv, summary_csv, report_md


def _runs_from_checkpoint(
    *,
    root: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    metric_names: list[str],
    include_failed: bool,
) -> list[CollectedRun]:
    rows: list[CollectedRun] = []
    experiment = _experiment_name(root, checkpoint_path)
    for run_id, record in checkpoint.get("runs", {}).items():
        status = str(record.get("status", "unknown"))
        if status != "completed" and not include_failed:
            continue
        payload = record.get("payload", {})
        recipe = payload.get("recipe", {})
        model_name = str(recipe.get("model_name") or payload.get("variant") or "unknown")
        recipe_id = recipe.get("recipe_id") or payload.get("run_id")
        metrics_raw = payload.get("metrics", {})
        metrics_raw = metrics_raw if isinstance(metrics_raw, dict) else {}
        metrics = _numeric_metrics(metrics_raw, metric_names)
        diagnosis = payload.get("diagnosis") if isinstance(payload.get("diagnosis"), dict) else {}
        rows.append(
            CollectedRun(
                run_id=str(run_id),
                experiment=experiment,
                checkpoint=str(checkpoint_path),
                stage=str(record.get("stage", recipe.get("stage", "unknown"))),
                status=status,
                model_name=model_name,
                fold=_fold_from_run(str(run_id), str(recipe_id) if recipe_id else None, payload.get("fold")),
                metrics=metrics,
                metrics_path=_none_or_str(metrics_raw.get("metrics_path")),
                config_path=_none_or_str(payload.get("config_path")),
                stdout_path=_none_or_str(payload.get("stdout_path")),
                log_dir=_none_or_str(payload.get("log_dir")),
                error=_none_or_str(payload.get("error")),
                diagnosis_category=_none_or_str(diagnosis.get("category")),
                diagnosis_summary=_none_or_str(diagnosis.get("summary")),
            )
        )
    return rows


def _numeric_metrics(metrics: dict[str, Any], metric_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in metric_names:
        value = metrics.get(name)
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _fold_from_run(run_id: str, recipe_id: str | None, payload_fold: Any) -> int | None:
    try:
        if payload_fold is not None:
            return int(payload_fold)
    except (TypeError, ValueError):
        pass
    for value in (recipe_id, run_id):
        if not value:
            continue
        match = FOLD_RE.search(value)
        if match:
            return int(match.group("fold"))
    return None


def _experiment_name(root: Path, checkpoint_path: Path) -> str:
    try:
        relative = checkpoint_path.parent.relative_to(root)
    except ValueError:
        relative = checkpoint_path.parent
    text = str(relative).replace("\\", "/")
    return text if text not in {"", "."} else checkpoint_path.parent.name


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_runs_csv(bundle: ResultBundle, path: Path) -> None:
    metric_names = list(bundle.metrics)
    fieldnames = [
        "run_id",
        "experiment",
        "checkpoint",
        "stage",
        "status",
        "model_name",
        "fold",
        *metric_names,
        "metrics_path",
        "config_path",
        "stdout_path",
        "log_dir",
        "error",
        "diagnosis_category",
        "diagnosis_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in bundle.runs:
            row = asdict(run)
            metrics = row.pop("metrics")
            row.update({metric: metrics.get(metric) for metric in metric_names})
            writer.writerow(row)


def _write_summary_csv(bundle: ResultBundle, path: Path) -> None:
    metric_names = list(bundle.metrics)
    fieldnames = ["experiment", "model_name", "n_completed", "n_total"]
    for metric in metric_names:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in bundle.summaries:
            row: dict[str, Any] = {
                "experiment": summary.experiment,
                "model_name": summary.model_name,
                "n_completed": summary.n_completed,
                "n_total": summary.n_total,
            }
            for metric in metric_names:
                values = summary.metrics.get(metric, {})
                row[f"{metric}_mean"] = values.get("mean")
                row[f"{metric}_std"] = values.get("std")
                row[f"{metric}_min"] = values.get("min")
                row[f"{metric}_max"] = values.get("max")
            writer.writerow(row)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _summary_sort_value(summary: ModelResultSummary, metric: str) -> float:
    value = summary.metrics.get(metric, {}).get("mean")
    return float(value) if value is not None else float("-inf")


def _fold_sort_value(run: CollectedRun) -> int:
    return run.fold if run.fold is not None else -1


def _render_markdown(bundle: ResultBundle) -> str:
    ranked = sorted(
        bundle.summaries,
        key=lambda item: _summary_sort_value(item, bundle.primary_metric),
        reverse=True,
    )
    lines = [
        "# Manuscript Results Index",
        "",
        f"- Generated: `{bundle.generated_at}`",
        f"- Root: `{bundle.root}`",
        f"- Checkpoints: `{len(bundle.checkpoints)}`",
        f"- Runs: `{len(bundle.runs)}`",
        f"- Primary metric: `{bundle.primary_metric}`",
        "",
        "## Primary Metric Ranking",
        "",
        "| Rank | Experiment | Model/Variant | Completed | Primary metric mean +/- std | Range |",
        "|---:|---|---|---:|---:|---|",
    ]
    for rank, row in enumerate(ranked, start=1):
        metric = row.metrics.get(bundle.primary_metric, {})
        mean_value = metric.get("mean")
        std_value = metric.get("std")
        score = "" if mean_value is None else f"{mean_value:.4f} +/- {std_value or 0.0:.4f}"
        range_text = ""
        if metric.get("min") is not None:
            range_text = f"{_fmt(metric.get('min'))}-{_fmt(metric.get('max'))}"
        lines.append(
            f"| {rank} | {row.experiment} | {row.model_name} | {row.n_completed}/{row.n_total} | "
            f"{score} | {range_text} |"
        )

    lines.extend(["", "## Run Inventory", ""])
    lines.append("| Experiment | Run | Model/Variant | Fold | Stage | Status | Primary metric | Artifact |")
    lines.append("|---|---|---|---:|---|---|---:|---|")
    for run in sorted(bundle.runs, key=lambda item: (item.experiment, item.model_name, _fold_sort_value(item), item.run_id)):
        metric = run.metrics.get(bundle.primary_metric)
        artifact = run.metrics_path or run.log_dir or run.stdout_path or ""
        lines.append(
            f"| {run.experiment} | `{run.run_id}` | {run.model_name} | {run.fold if run.fold is not None else ''} | "
            f"{run.stage} | {run.status} | {_fmt(metric)} | `{artifact}` |"
        )

    if bundle.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in bundle.warnings])
    return "\n".join(lines)
