from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .data import prepare_dataset_kfold
from .innovation_cv import (
    CustomRunResult,
    custom_result_from_payload,
    custom_result_to_payload,
    run_custom_variant,
    summarize,
)
from .split_executor import materialize_kfold_from_split_plan, select_split_plan
from .state import ExperimentCheckpoint, ResearchJournal


@dataclass(frozen=True)
class AblationVariant:
    variant: str
    focal_loss: bool
    prototype_head: bool
    description: str


DEFAULT_ABLATIONS = [
    AblationVariant("AB_MIL_CE", False, False, "Baseline AB_MIL with cross entropy."),
    AblationVariant("AB_MIL_FOCAL", True, False, "Add class-balanced focal loss only."),
    AblationVariant("AB_MIL_PROTO", False, True, "Add prototype auxiliary head only."),
    AblationVariant("AB_MIL_FOCAL_PROTO", True, True, "Full method: focal loss plus prototype head."),
]


def _variant_map(variants: list[str] | None = None) -> list[AblationVariant]:
    known = {item.variant: item for item in DEFAULT_ABLATIONS}
    if variants is None:
        return DEFAULT_ABLATIONS
    out = []
    for variant in variants:
        if variant in known:
            out.append(known[variant])
        else:
            out.append(
                AblationVariant(
                    variant=variant,
                    focal_loss="FOCAL" in variant.upper(),
                    prototype_head="PROTO" in variant.upper(),
                    description="Custom ablation variant inferred from its name.",
                )
            )
    return out


def _write_ablation_report(
    output_dir: Path,
    variants: list[AblationVariant],
    results: list[CustomRunResult],
    summary: dict[str, dict[str, Any]],
) -> Path:
    report = output_dir / "ablation_cv_report.md"
    variant_meta = {item.variant: item for item in variants}
    ranked = sorted(
        summary.items(),
        key=lambda item: item[1].get("test_macro_auc_mean", float("-inf")),
        reverse=True,
    )
    lines = [
        "# Ablation CV Report",
        "",
        "## Matrix",
        "",
        "| Variant | Focal loss | Prototype head | Description |",
        "|---|---|---|---|",
    ]
    for item in variants:
        lines.append(f"| {item.variant} | {item.focal_loss} | {item.prototype_head} | {item.description} |")
    lines.extend(
        [
            "",
            "## Mean Metrics",
            "",
            "| Rank | Variant | Focal | Prototype | Completed | Test macro AUC | Test BACC | Test macro F1 | Test ACC | Val macro AUC |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, (variant, row) in enumerate(ranked, start=1):
        meta = variant_meta.get(variant)

        def fmt(name: str) -> str:
            m = row.get(f"{name}_mean")
            s = row.get(f"{name}_std")
            if m is None:
                return ""
            return f"{m:.4f} +/- {s:.4f}"

        lines.append(
            f"| {rank} | {variant} | {getattr(meta, 'focal_loss', '')} | {getattr(meta, 'prototype_head', '')} | "
            f"{row.get('n_completed', 0)}/{row.get('n_total', 0)} | {fmt('test_macro_auc')} | "
            f"{fmt('test_bacc')} | {fmt('test_macro_f1')} | {fmt('test_acc')} | {fmt('val_macro_auc')} |"
        )
    lines.extend(["", "## Per-Fold Runs", ""])
    lines.append("| Run | Variant | Fold | Status | Diagnosis | Test macro AUC | Metrics |")
    lines.append("|---|---|---:|---|---|---:|---|")
    for result in results:
        auc = result.metrics.get("test_macro_auc")
        auc_text = "" if auc is None else f"{float(auc):.4f}"
        diagnosis = ""
        if result.diagnosis is not None:
            diagnosis = f"{result.diagnosis.category}: {result.diagnosis.summary}".replace("|", "/")
        lines.append(
            f"| `{result.run_id}` | {result.variant} | {result.fold} | {result.status} | "
            f"{diagnosis} | {auc_text} | `{result.metrics.get('metrics_path', '')}` |"
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_ablation_cv(
    cfg: AutoMilConfig,
    *,
    variants: list[str] | None,
    n_splits: int,
    epochs: int,
    split_plan_path: Path | None = None,
    split_plan_id: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    output_dir = cfg.output_dir / "ablation_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")
    task = cfg.task_spec
    dataset = cfg.dataset_spec
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    selected_split = None
    if split_plan_path is not None:
        selected_split = select_split_plan(split_plan_path, split_plan_id)
        artifacts = materialize_kfold_from_split_plan(
            dataset=dataset,
            task=task,
            output_dir=output_dir / "folds",
            split_plan_path=split_plan_path,
            plan_id=split_plan_id,
        )
        n_splits = int(selected_split.plan.get("n_splits", n_splits))
    else:
        artifacts = prepare_dataset_kfold(dataset=dataset, task=task, output_dir=output_dir / "folds", n_splits=n_splits)
    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
    checkpoint.update_metadata(
        command="run-ablation-cv",
        config=str(cfg.path),
        n_splits=n_splits,
        metadata_json=str(artifacts.metadata_json),
        dry_run=dry_run,
        split_plan=str(split_plan_path) if split_plan_path else None,
        split_plan_id=selected_split.plan_id if selected_split else None,
    )
    in_dim_cfg = training.get("in_dim", "auto")
    in_dim = int(metadata["feature"]["in_dim"] if in_dim_cfg == "auto" else in_dim_cfg)
    lr = float(search.get("learning_rates", [0.0002])[0])
    dropout = 0.25
    matrix = _variant_map(variants)
    results: list[CustomRunResult] = []
    for fold_idx, fold_dir in enumerate(artifacts.fold_dirs):
        dataset_csv = fold_dir / "dataset.csv"
        for item in matrix:
            run_id = f"{item.variant.lower()}_fold{fold_idx}"
            cached_payload = checkpoint.get_completed_payload(run_id) if resume and not dry_run else None
            if cached_payload is not None:
                result = custom_result_from_payload(cached_payload)
                results.append(result)
                journal.write("ablation_run_resume", {"fold": fold_idx, "variant": item.variant, "run_id": run_id})
                continue
            journal.write(
                "ablation_run_start",
                {"fold": fold_idx, "variant": asdict(item), "dataset_csv": str(dataset_csv)},
            )
            result = run_custom_variant(
                cfg=cfg,
                dataset_csv=dataset_csv,
                output_dir=output_dir,
                variant=item.variant,
                fold=fold_idx,
                num_classes=int(metadata["num_classes"]),
                in_dim=in_dim,
                epochs=epochs,
                lr=lr,
                dropout=dropout,
                balanced_sampler=True,
                prototype_lambda=0.2,
                dry_run=dry_run,
            )
            results.append(result)
            payload = custom_result_to_payload(result)
            checkpoint.record_run(result.run_id, "ablation_cv", result.status, payload)
            journal.write("ablation_run_result", payload)

    summary = summarize(results)
    summary_json = output_dir / "ablation_cv_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _write_ablation_report(output_dir, matrix, results, summary)
    checkpoint.update_metadata(report=str(report), summary_json=str(summary_json))
    journal.write("ablation_report", {"path": str(report), "summary_json": str(summary_json)})
    return report
