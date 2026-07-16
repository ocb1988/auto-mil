from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .state import json_ready, now_iso


DEFAULT_PRIMARY_METRIC = "test_macro_auc"


@dataclass(frozen=True)
class EvidenceIndex:
    root: str
    metadata_json: str | None = None
    manuscript_results: str | None = None
    model_summary_csv: str | None = None
    stats_report: str | None = None
    prediction_report: str | None = None
    figure_report: str | None = None
    ablation_report: str | None = None
    innovation_report: str | None = None
    innovation_summary_json: str | None = None
    baseline_plan: str | None = None
    split_plan: str | None = None


@dataclass(frozen=True)
class ManuscriptDraft:
    generated_at: str
    title: str
    root: str
    primary_metric: str
    evidence: EvidenceIndex
    warnings: list[str] = field(default_factory=list)


def write_manuscript_draft(
    root: str | Path,
    *,
    cfg: AutoMilConfig | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> tuple[ManuscriptDraft, Path, Path]:
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / "manuscript"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = discover_evidence(root)
    warnings: list[str] = []
    metadata = _load_json(Path(evidence.metadata_json)) if evidence.metadata_json else {}
    if not metadata:
        warnings.append("No metadata.json was found; dataset Methods text is incomplete.")
    model_rows = _read_csv(Path(evidence.model_summary_csv)) if evidence.model_summary_csv else []
    if not model_rows:
        warnings.append("No model_summary.csv was found; main comparison table is incomplete.")
    completed_rows = [row for row in model_rows if _to_int(row.get("n_completed")) > 0]
    if model_rows and not completed_rows:
        warnings.append("Model summary exists but has no completed runs; Results should be treated as a dry-run scaffold.")

    draft = ManuscriptDraft(
        generated_at=now_iso(),
        title=title or _default_title(cfg, metadata),
        root=str(root),
        primary_metric=primary_metric,
        evidence=evidence,
        warnings=warnings,
    )
    json_path = output_dir / "manuscript_evidence.json"
    md_path = output_dir / "manuscript_draft.md"
    json_path.write_text(json.dumps(json_ready(asdict(draft)), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(
        _render_draft(draft, cfg=cfg, metadata=metadata, model_rows=model_rows),
        encoding="utf-8",
    )
    return draft, json_path, md_path


def discover_evidence(root: str | Path) -> EvidenceIndex:
    root = Path(root)
    return EvidenceIndex(
        root=str(root),
        metadata_json=_first_existing(
            root,
            [
                "case_level_cv/folds/metadata.json",
                "experiment_tree/dataset/metadata.json",
                "metadata.json",
            ],
            fallback_glob="**/metadata.json",
        ),
        manuscript_results=_first_existing(root, ["results/manuscript_results.md"]),
        model_summary_csv=_first_existing(root, ["results/model_summary.csv"]),
        stats_report=_first_existing(root, ["case_level_cv/stats/stats_report.md", "stats/stats_report.md"], fallback_glob="**/stats_report.md"),
        prediction_report=_first_existing(root, ["prediction_aggregation/prediction_report.md"]),
        figure_report=_first_existing(root, ["prediction_aggregation/figures/figure_report.md"]),
        ablation_report=_first_existing(root, ["ablation_cv/ablation_cv_report.md"]),
        innovation_report=_first_existing(root, ["innovation_cv/innovation_cv_report.md"]),
        innovation_summary_json=_first_existing(root, ["innovation_cv/innovation_cv_summary.json"]),
        baseline_plan=_first_existing(root, ["baseline_plan/baseline_plan.md"]),
        split_plan=_first_existing(root, ["split_plan/split_plan.md"]),
    )


def _first_existing(root: Path, relative_paths: list[str], fallback_glob: str | None = None) -> str | None:
    for rel in relative_paths:
        path = root / rel
        if path.exists():
            return str(path)
    if fallback_glob:
        matches = sorted(root.glob(fallback_glob), key=lambda path: (len(path.parts), str(path)))
        if matches:
            return str(matches[0])
    return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except OSError:
        return []


def _default_title(cfg: AutoMilConfig | None, metadata: dict[str, Any]) -> str:
    dataset = metadata.get("dataset") or (cfg.dataset_spec.name if cfg else "Pathology dataset")
    label = metadata.get("label_column") or (cfg.task_spec.label_column if cfg else "endpoint")
    return f"Autonomous MIL experiments for {dataset} {label} prediction"


def _to_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        if value in {None, ""}:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _best_row(rows: list[dict[str, str]], primary_metric: str) -> dict[str, str] | None:
    metric_key = f"{primary_metric}_mean"
    scored = [(row, _to_float(row.get(metric_key))) for row in rows]
    scored = [(row, score) for row, score in scored if score is not None and _to_int(row.get("n_completed")) > 0]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def _metric_text(row: dict[str, str] | None, primary_metric: str) -> str:
    if not row:
        return "not yet available"
    mean = _to_float(row.get(f"{primary_metric}_mean"))
    std = _to_float(row.get(f"{primary_metric}_std"))
    if mean is None:
        return "not yet available"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _class_map_text(metadata: dict[str, Any]) -> str:
    label_to_id = metadata.get("label_to_id") or {}
    if not isinstance(label_to_id, dict) or not label_to_id:
        return "not available"
    return ", ".join(f"{name}={idx}" for name, idx in sorted(label_to_id.items(), key=lambda item: item[1]))


def _counts_text(metadata: dict[str, Any], key: str) -> str:
    counts = metadata.get(key)
    if not counts:
        return "not available"
    return json.dumps(counts, ensure_ascii=False, sort_keys=True)


def _artifact_line(label: str, path: str | None) -> str:
    return f"- {label}: `{path}`" if path else f"- {label}: not available"


def _result_table(rows: list[dict[str, str]], primary_metric: str) -> list[str]:
    if not rows:
        return ["No model summary table was available."]
    metric_mean = f"{primary_metric}_mean"
    metric_std = f"{primary_metric}_std"
    ranked = sorted(
        rows,
        key=lambda row: _to_float(row.get(metric_mean)) if _to_float(row.get(metric_mean)) is not None else float("-inf"),
        reverse=True,
    )
    lines = [
        f"| Rank | Experiment | Model | Completed | {primary_metric} |",
        "|---:|---|---|---:|---:|",
    ]
    for idx, row in enumerate(ranked, start=1):
        completed = f"{_to_int(row.get('n_completed'))}/{_to_int(row.get('n_total'))}"
        metric = _metric_text(row, primary_metric)
        lines.append(f"| {idx} | {row.get('experiment', '')} | {row.get('model_name', '')} | {completed} | {metric} |")
    return lines


def _read_excerpt(path: str | None, max_lines: int = 24) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[:max_lines]


def _render_draft(
    draft: ManuscriptDraft,
    *,
    cfg: AutoMilConfig | None,
    metadata: dict[str, Any],
    model_rows: list[dict[str, str]],
) -> str:
    task = cfg.task_spec if cfg else None
    dataset = cfg.dataset_spec if cfg else None
    best = _best_row(model_rows, draft.primary_metric)
    evidence = draft.evidence
    lines = [
        f"# {draft.title}",
        "",
        "## Evidence Status",
        "",
        f"- Generated: `{draft.generated_at}`",
        f"- Run root: `{draft.root}`",
        _artifact_line("Dataset metadata", evidence.metadata_json),
        _artifact_line("Main result index", evidence.manuscript_results),
        _artifact_line("Model summary table", evidence.model_summary_csv),
        _artifact_line("Statistical report", evidence.stats_report),
        _artifact_line("Prediction report", evidence.prediction_report),
        _artifact_line("Figure report", evidence.figure_report),
        _artifact_line("Innovation report", evidence.innovation_report),
        _artifact_line("Innovation summary", evidence.innovation_summary_json),
        _artifact_line("Ablation report", evidence.ablation_report),
        "",
        "## Methods",
        "",
        "### Dataset and Task",
        "",
        (
            f"We evaluated multiple-instance learning models on `{metadata.get('dataset', dataset.name if dataset else 'the dataset')}` "
            f"for `{metadata.get('label_column', task.label_column if task else 'the endpoint')}` prediction. "
            f"The executable task type was `{task.kind if task else metadata.get('task', {}).get('kind', 'classification')}`."
        ),
        "",
        (
            f"The analysis included `{metadata.get('num_cases', 'NA')}` cases and "
            f"`{metadata.get('num_h5_matched', 'NA')}` matched source slide feature files. "
            f"Class mapping was: {_class_map_text(metadata)}."
        ),
        "",
        "### Bag Construction",
        "",
        (
            f"The default prediction unit was patient/case. The run metadata records bag level "
            f"`{metadata.get('bag_level', dataset.bag_level if dataset else 'case')}`. "
            "For case-level MIL, patch features from all slides belonging to the same case were concatenated before training, "
            "and source slide provenance was retained in the case-bag manifest."
        ),
        "",
        "### Split Design",
        "",
        (
            "Splits were performed at the patient/case level to avoid slide-level leakage. "
            f"Case split counts were `{_counts_text(metadata, 'split_counts_cases')}`. "
            f"Slide split counts were `{_counts_text(metadata, 'split_counts_slides')}`."
        ),
        "",
        "### Baselines and Proposed Method",
        "",
        (
            "Baseline selection followed a manuscript-oriented MIL suite including classic attention MIL, transformer/long-context MIL, "
            "RRT-style MIL, and recent MIL methods when compatible with the local runtime. The proposed method and ablation variants "
            "were run under the same split policy and metric contract as the baselines. Method-track changes were separated from "
            "support-track changes such as ensembling, thresholding, seed search, or hyperparameter-only tuning."
        ),
        "",
        "### Evaluation",
        "",
        (
            f"The primary metric was `{draft.primary_metric}`. Completed fold or seed runs were summarized as mean +/- standard deviation, "
            "with confidence intervals and paired comparisons reported when common folds were available. Case-level prediction artifacts "
            "were used for ROC, confusion matrix, calibration, per-class performance, and error-case analysis."
        ),
        "",
        "## Results",
        "",
        "### Main Comparison",
        "",
        *(_result_table(model_rows, draft.primary_metric)),
        "",
        (
            f"The best completed row by `{draft.primary_metric}` was "
            f"`{best.get('model_name') if best else 'not available'}` from `{best.get('experiment') if best else 'not available'}` "
            f"with `{_metric_text(best, draft.primary_metric)}`. Treat this statement as provisional unless the evidence status above "
            "shows completed training runs and the statistical report is available."
        ),
        "",
        "### Method Innovation Integrity",
        "",
    ]
    innovation_excerpt = _read_excerpt(evidence.innovation_report)
    if innovation_excerpt:
        lines.extend(innovation_excerpt)
    else:
        lines.append("No innovation-track report was available.")
    lines.extend(
        [
            "",
            "Method-track rows may be used to support the proposed algorithm only when their core modules are described and ablated. "
            "Support-track rows should be reported as auxiliary robustness, tuning, or upper-bound evidence rather than as the main method contribution.",
            "",
            "### Statistical Analysis",
            "",
        ]
    )
    stats_excerpt = _read_excerpt(evidence.stats_report)
    lines.extend(stats_excerpt or ["No statistical report was available."])
    lines.extend(["", "### Prediction Figures and Error Analysis", ""])
    figure_excerpt = _read_excerpt(evidence.figure_report)
    prediction_excerpt = _read_excerpt(evidence.prediction_report, max_lines=16)
    lines.extend(figure_excerpt or prediction_excerpt or ["No case-level prediction figure report was available."])
    lines.extend(["", "### Ablation Study", ""])
    ablation_excerpt = _read_excerpt(evidence.ablation_report)
    lines.extend(ablation_excerpt or ["No ablation report was available."])
    lines.extend(["", "## Limitations and Evidence Gaps", ""])
    gaps = list(draft.warnings)
    if not evidence.stats_report:
        gaps.append("Statistical confidence intervals or paired tests are not yet available.")
    if not evidence.figure_report:
        gaps.append("Case-level ROC/confusion/calibration figures are not yet available.")
    if not evidence.ablation_report:
        gaps.append("Ablation evidence is not yet available.")
    if not gaps:
        gaps.append("No major artifact gaps were detected by the manuscript writer.")
    lines.extend([f"- {gap}" for gap in gaps])
    lines.extend(
        [
            "",
            "## Claim Guardrails",
            "",
            "- Do not claim state-of-the-art performance unless the baseline suite, split policy, and external validation match the target literature.",
            "- Do not claim clinical deployability without external validation, calibration analysis, and prospective or multi-center evidence.",
            "- Do not describe support-track improvements such as ensemble, seed search, threshold tuning, or hyperparameter-only recipes as the main algorithmic contribution.",
            "- Treat dry-run, failed, or incomplete experiments as workflow validation only.",
        ]
    )
    return "\n".join(lines)
