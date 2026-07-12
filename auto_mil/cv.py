from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import AutoMilConfig
from .data import prepare_cptac_brca_kfold
from .mil_baseline import Recipe, RunResult, run_recipe, run_result_from_payload, run_result_to_payload
from .state import ExperimentCheckpoint, ResearchJournal


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recipe_id(fold_idx: int, model: str) -> str:
    return f"cv_fold{fold_idx}_{model.lower()}"


def build_cv_recipes(
    models: list[str],
    *,
    fold_idx: int,
    epochs: int,
    lr: float,
    dropout: float,
    balanced_sampler: bool,
) -> list[Recipe]:
    recipes: list[Recipe] = []
    for model in models:
        recipes.append(
            Recipe(
                recipe_id=_recipe_id(fold_idx, model),
                stage="baseline_cv",
                model_name=model,
                epochs=epochs,
                lr=lr,
                dropout=dropout,
                balanced_sampler=balanced_sampler,
                notes=f"Case-level CV baseline fold {fold_idx}.",
            )
        )
    return recipes


def summarize_cv(results: list[RunResult]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[RunResult]] = {}
    for result in results:
        by_model.setdefault(result.recipe.model_name, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    metric_names = ["test_macro_auc", "test_bacc", "test_macro_f1", "test_acc", "val_macro_auc"]
    for model, model_results in sorted(by_model.items()):
        row: dict[str, Any] = {
            "n_completed": sum(1 for r in model_results if r.status == "completed"),
            "n_total": len(model_results),
        }
        for metric in metric_names:
            values = []
            for result in model_results:
                value = result.metrics.get(metric)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            if values:
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary[model] = row
    return summary


def _write_summary_csv(summary: dict[str, dict[str, Any]], path: Path) -> None:
    keys = sorted({key for row in summary.values() for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *keys])
        writer.writeheader()
        for model, row in summary.items():
            writer.writerow({"model": model, **row})


def _write_cv_report(
    output_dir: Path,
    cfg: AutoMilConfig,
    metadata: dict[str, Any],
    results: list[RunResult],
    summary: dict[str, dict[str, Any]],
) -> Path:
    report_path = output_dir / "cv_report.md"
    ranked = sorted(
        summary.items(),
        key=lambda item: item[1].get("test_macro_auc_mean", float("-inf")),
        reverse=True,
    )
    lines = [
        f"# Case-Level CV Report: {cfg.name}",
        "",
        f"- Dataset: `{metadata.get('dataset')}`",
        f"- Label: `{metadata.get('label_column')}`",
        f"- Cases: `{metadata.get('num_cases')}`",
        f"- Slides: `{metadata.get('num_h5_matched')}`",
        f"- Folds: `{metadata.get('n_splits')}`",
        f"- Feature dim: `{metadata.get('feature', {}).get('in_dim')}`",
        "",
        "## Mean Metrics",
        "",
        "| Rank | Model | Completed | Test macro AUC | Test BACC | Test macro F1 | Test ACC | Val macro AUC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (model, row) in enumerate(ranked, start=1):
        def fmt(name: str) -> str:
            m = row.get(f"{name}_mean")
            s = row.get(f"{name}_std")
            if m is None:
                return ""
            return f"{m:.4f} +/- {s:.4f}"

        lines.append(
            f"| {rank} | {model} | {row.get('n_completed', 0)}/{row.get('n_total', 0)} | "
            f"{fmt('test_macro_auc')} | {fmt('test_bacc')} | {fmt('test_macro_f1')} | "
            f"{fmt('test_acc')} | {fmt('val_macro_auc')} |"
        )

    lines.extend(["", "## Per-Fold Runs", ""])
    lines.append("| Fold/Recipe | Model | Status | Diagnosis | Test macro AUC | Metrics |")
    lines.append("|---|---|---|---|---:|---|")
    for result in results:
        auc = result.metrics.get("test_macro_auc")
        auc_text = "" if auc is None else f"{float(auc):.4f}"
        diagnosis = ""
        if result.diagnosis is not None:
            diagnosis = f"{result.diagnosis.category}: {result.diagnosis.summary}".replace("|", "/")
        lines.append(
            f"| `{result.recipe.recipe_id}` | {result.recipe.model_name} | {result.status} | {diagnosis} | "
            f"{auc_text} | `{result.metrics.get('metrics_path', '')}` |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_case_level_cv(
    cfg: AutoMilConfig,
    *,
    models: list[str] | None = None,
    n_splits: int = 5,
    epochs: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    output_dir = cfg.output_dir / "case_level_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")
    task = cfg.raw.get("task", {})
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})

    artifacts = prepare_cptac_brca_kfold(
        data_dir=cfg.data_dir,
        labels_csv=cfg.labels_csv,
        output_dir=output_dir / "folds",
        label_column=str(task.get("label_column", "PAM50")),
        min_class_count=int(task.get("min_class_count", 2)),
        seed=int(task.get("split_seed", 2024)),
        n_splits=n_splits,
        val_fraction_of_train=float(task.get("cv_val_fraction_of_train", 0.2)),
    )
    metadata = _load_json(artifacts.metadata_json)
    checkpoint.update_metadata(
        command="run-cv",
        config=str(cfg.path),
        n_splits=n_splits,
        metadata_json=str(artifacts.metadata_json),
        dry_run=dry_run,
    )
    journal.write("cv_dataset_prepared", {"metadata_json": str(artifacts.metadata_json), "metadata": metadata})

    selected_models = models or list(
        search.get("cv_models", ["MEAN_MIL", "MAX_MIL", "AB_MIL", "GATE_AB_MIL", "FR_MIL"])
    )
    recipe_epochs = int(epochs if epochs is not None else training.get("cv_epochs", 3))
    lr = float(search.get("learning_rates", [0.0002])[0])
    dropout = float(search.get("dropouts", [0.1])[0])
    in_dim_cfg = training.get("in_dim", "auto")
    in_dim = int(metadata["feature"]["in_dim"] if in_dim_cfg == "auto" else in_dim_cfg)

    results: list[RunResult] = []
    for fold_idx, fold_dir in enumerate(artifacts.fold_dirs):
        dataset_csv = fold_dir / "dataset.csv"
        fold_recipes = build_cv_recipes(
            selected_models,
            fold_idx=fold_idx,
            epochs=recipe_epochs,
            lr=lr,
            dropout=dropout,
            balanced_sampler=bool(training.get("cv_balanced_sampler", False)),
        )
        for recipe in fold_recipes:
            cached_payload = checkpoint.get_completed_payload(recipe.recipe_id) if resume and not dry_run else None
            if cached_payload is not None:
                result = run_result_from_payload(cached_payload)
                results.append(result)
                journal.write(
                    "cv_run_resume",
                    {"fold": fold_idx, "recipe": asdict(recipe), "status": result.status},
                )
                continue
            journal.write("cv_run_start", {"fold": fold_idx, "recipe": asdict(recipe), "dataset_csv": str(dataset_csv)})
            result = run_recipe(
                recipe,
                python=cfg.python,
                mil_baseline_dir=cfg.mil_baseline_dir,
                dataset_name=f"{cfg.name}_fold{fold_idx}",
                dataset_csv_path=dataset_csv,
                output_dir=output_dir,
                num_classes=int(metadata["num_classes"]),
                in_dim=in_dim,
                device=int(training.get("device", 0)),
                num_workers=int(training.get("num_workers", 0)),
                best_metric=str(training.get("best_metric", "macro_auc")),
                dry_run=dry_run,
            )
            results.append(result)
            payload = run_result_to_payload(result)
            checkpoint.record_run(recipe.recipe_id, recipe.stage, result.status, payload)
            journal.write("cv_run_result", payload)

    summary = summarize_cv(results)
    summary_json = output_dir / "cv_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(summary, output_dir / "cv_summary.csv")
    report = _write_cv_report(output_dir, cfg, metadata, results, summary)
    checkpoint.update_metadata(report=str(report), summary_json=str(summary_json))
    journal.write("cv_report", {"path": str(report), "summary_json": str(summary_json)})
    return report
