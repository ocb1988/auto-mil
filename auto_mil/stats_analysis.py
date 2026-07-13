from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .state import json_ready, now_iso


FOLD_RE = re.compile(r"(?:^|_)fold(?P<fold>\d+)(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class FoldMetric:
    run_id: str
    model_name: str
    fold: int | None
    metric: str
    value: float


@dataclass(frozen=True)
class ModelMetricSummary:
    model_name: str
    n: int
    mean: float | None
    std: float | None
    ci95_low: float | None
    ci95_high: float | None
    values_by_fold: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseComparison:
    model_name: str
    baseline_model: str
    n_common_folds: int
    mean_diff: float | None
    ci95_low: float | None
    ci95_high: float | None
    paired_t_pvalue: float | None
    wilcoxon_pvalue: float | None


@dataclass(frozen=True)
class StatsReport:
    checkpoint: str
    metric: str
    generated_at: str
    summaries: list[ModelMetricSummary]
    comparisons: list[PairwiseComparison]
    warnings: list[str] = field(default_factory=list)


def _critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96


def _ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    center = mean(values)
    margin = _critical_95(len(values)) * stdev(values) / math.sqrt(len(values))
    return center - margin, center + margin


def _pvalues(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(b) < 2:
        return None, None
    try:
        from scipy import stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                w_p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            w_p = None
        return t_p, w_p
    except Exception:
        return None, None


def _fold_from_run(run_id: str, recipe_id: str | None = None) -> int | None:
    for value in (recipe_id, run_id):
        if not value:
            continue
        match = FOLD_RE.search(value)
        if match:
            return int(match.group("fold"))
    return None


def extract_fold_metrics(checkpoint_path: str | Path, metric: str) -> list[FoldMetric]:
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    rows: list[FoldMetric] = []
    for run_id, record in checkpoint.get("runs", {}).items():
        if record.get("status") != "completed":
            continue
        payload = record.get("payload", {})
        metrics = payload.get("metrics", {})
        if metric not in metrics:
            continue
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        recipe = payload.get("recipe", {})
        model_name = str(recipe.get("model_name", payload.get("variant", "unknown")))
        recipe_id = recipe.get("recipe_id")
        rows.append(
            FoldMetric(
                run_id=str(run_id),
                model_name=model_name,
                fold=_fold_from_run(str(run_id), str(recipe_id) if recipe_id else None),
                metric=metric,
                value=value,
            )
        )
    return rows


def summarize_fold_metrics(rows: list[FoldMetric]) -> list[ModelMetricSummary]:
    by_model: dict[str, list[FoldMetric]] = {}
    for row in rows:
        by_model.setdefault(row.model_name, []).append(row)
    summaries: list[ModelMetricSummary] = []
    for model, model_rows in sorted(by_model.items()):
        values = [row.value for row in model_rows]
        low, high = _ci95(values)
        summaries.append(
            ModelMetricSummary(
                model_name=model,
                n=len(values),
                mean=mean(values) if values else None,
                std=stdev(values) if len(values) > 1 else 0.0 if values else None,
                ci95_low=low,
                ci95_high=high,
                values_by_fold={str(row.fold if row.fold is not None else row.run_id): row.value for row in model_rows},
            )
        )
    return summaries


def compare_to_baseline(rows: list[FoldMetric], baseline_model: str) -> list[PairwiseComparison]:
    values: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.fold is None:
            continue
        values.setdefault(row.model_name, {})[row.fold] = row.value
    baseline = values.get(baseline_model, {})
    comparisons: list[PairwiseComparison] = []
    for model, model_values in sorted(values.items()):
        if model == baseline_model:
            continue
        common = sorted(set(model_values) & set(baseline))
        diffs = [model_values[fold] - baseline[fold] for fold in common]
        low, high = _ci95(diffs)
        t_p, w_p = _pvalues([model_values[fold] for fold in common], [baseline[fold] for fold in common])
        comparisons.append(
            PairwiseComparison(
                model_name=model,
                baseline_model=baseline_model,
                n_common_folds=len(common),
                mean_diff=mean(diffs) if diffs else None,
                ci95_low=low,
                ci95_high=high,
                paired_t_pvalue=t_p,
                wilcoxon_pvalue=w_p,
            )
        )
    return comparisons


def build_stats_report(checkpoint_path: str | Path, metric: str, baseline_model: str | None = None) -> StatsReport:
    rows = extract_fold_metrics(checkpoint_path, metric)
    warnings = []
    if not rows:
        warnings.append(f"No completed runs with metric {metric!r} were found.")
    summaries = summarize_fold_metrics(rows)
    comparisons = compare_to_baseline(rows, baseline_model) if baseline_model else []
    if baseline_model and not any(row.model_name == baseline_model for row in rows):
        warnings.append(f"Baseline model {baseline_model!r} was not found among completed runs.")
    return StatsReport(
        checkpoint=str(checkpoint_path),
        metric=metric,
        generated_at=now_iso(),
        summaries=summaries,
        comparisons=comparisons,
        warnings=warnings,
    )


def stats_report_to_payload(report: StatsReport) -> dict[str, Any]:
    return json_ready(asdict(report))


def write_stats_report(report: StatsReport, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stats_report.json"
    md_path = output_dir / "stats_report.md"
    json_path.write_text(json.dumps(stats_report_to_payload(report), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _render_markdown(report: StatsReport) -> str:
    lines = [
        "# Statistical Report",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Checkpoint: `{report.checkpoint}`",
        f"- Metric: `{report.metric}`",
        "",
        "## Model Summary",
        "",
        "| Model | N | Mean | Std | 95% CI | Fold values |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report.summaries:
        ci = f"[{_fmt(row.ci95_low)}, {_fmt(row.ci95_high)}]" if row.ci95_low is not None else ""
        lines.append(
            f"| {row.model_name} | {row.n} | {_fmt(row.mean)} | {_fmt(row.std)} | {ci} | `{row.values_by_fold}` |"
        )
    if report.comparisons:
        lines.extend(
            [
                "",
                "## Paired Comparisons",
                "",
                "| Model | Baseline | Common folds | Mean diff | 95% CI diff | Paired t p | Wilcoxon p |",
                "|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in report.comparisons:
            ci = f"[{_fmt(row.ci95_low)}, {_fmt(row.ci95_high)}]" if row.ci95_low is not None else ""
            lines.append(
                f"| {row.model_name} | {row.baseline_model} | {row.n_common_folds} | {_fmt(row.mean_diff)} | "
                f"{ci} | {_fmt(row.paired_t_pvalue)} | {_fmt(row.wilcoxon_pvalue)} |"
            )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in report.warnings])
    return "\n".join(lines)
