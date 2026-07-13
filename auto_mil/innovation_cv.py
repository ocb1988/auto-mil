from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import AutoMilConfig
from .data import prepare_dataset_kfold
from .log_analyzer import LogDiagnosis, diagnose_log_file, diagnosis_from_payload, write_diagnosis_json
from .split_executor import materialize_kfold_from_split_plan, select_split_plan
from .state import ExperimentCheckpoint, ResearchJournal, json_ready


@dataclass(frozen=True)
class CustomRunResult:
    run_id: str
    variant: str
    fold: int
    status: str
    command: list[str]
    stdout_path: Path
    log_dir: Path
    metrics: dict[str, Any]
    error: str | None = None
    diagnosis: LogDiagnosis | None = None


def custom_result_to_payload(result: CustomRunResult) -> dict[str, Any]:
    return json_ready(
        {
            "run_id": result.run_id,
            "variant": result.variant,
            "fold": result.fold,
            "status": result.status,
            "command": result.command,
            "stdout_path": result.stdout_path,
            "log_dir": result.log_dir,
            "metrics": result.metrics,
            "error": result.error,
            "diagnosis": result.diagnosis,
        }
    )


def custom_result_from_payload(payload: dict[str, Any]) -> CustomRunResult:
    return CustomRunResult(
        run_id=str(payload["run_id"]),
        variant=str(payload["variant"]),
        fold=int(payload["fold"]),
        status=str(payload["status"]),
        command=[str(x) for x in payload.get("command", [])],
        stdout_path=Path(payload["stdout_path"]),
        log_dir=Path(payload["log_dir"]),
        metrics=dict(payload.get("metrics", {})),
        error=payload.get("error"),
        diagnosis=diagnosis_from_payload(payload.get("diagnosis")),
    )


def _read_best_metrics(log_dir: Path) -> dict[str, Any]:
    candidates = sorted(log_dir.glob("Best_Log*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    path = candidates[-1]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"metrics_path": str(path)}
    metrics: dict[str, Any] = {"metrics_path": str(path)}
    for key, value in rows[0].items():
        if key.endswith("confusion_mat"):
            metrics[key] = value
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = value
    return metrics


def run_custom_variant(
    *,
    cfg: AutoMilConfig,
    dataset_csv: Path,
    output_dir: Path,
    variant: str,
    fold: int,
    num_classes: int,
    in_dim: int,
    epochs: int,
    lr: float,
    dropout: float,
    balanced_sampler: bool,
    prototype_lambda: float,
    dry_run: bool,
) -> CustomRunResult:
    run_id = f"{variant.lower()}_fold{fold}"
    stdout_dir = output_dir / "stdout"
    log_dir = output_dir / "mil_logs" / run_id
    stdout_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_dir / f"{run_id}.log"

    command = [
        str(cfg.python),
        "-B",
        "-m",
        "auto_mil.custom_abmil",
        "--dataset-csv",
        str(dataset_csv),
        "--mil-baseline-dir",
        str(cfg.mil_baseline_dir),
        "--output-dir",
        str(log_dir),
        "--variant",
        variant,
        "--num-classes",
        str(num_classes),
        "--in-dim",
        str(in_dim),
        "--epochs",
        str(epochs),
        "--device",
        str(cfg.raw.get("training", {}).get("device", 0)),
        "--lr",
        str(lr),
        "--dropout",
        str(dropout),
        "--prototype-lambda",
        str(prototype_lambda),
    ]
    if balanced_sampler:
        command.append("--balanced-sampler")

    if dry_run:
        stdout_path.write_text(json.dumps({"command": command}, indent=2), encoding="utf-8")
        return CustomRunResult(run_id, variant, fold, "dry_run", command, stdout_path, log_dir, {})

    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status = "completed" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"custom_abmil exited with code {proc.returncode}"
    diagnosis = None
    if proc.returncode != 0:
        diagnosis = diagnose_log_file(stdout_path)
        write_diagnosis_json(diagnosis, stdout_path.with_suffix(".diagnosis.json"))
    return CustomRunResult(
        run_id=run_id,
        variant=variant,
        fold=fold,
        status=status,
        command=command,
        stdout_path=stdout_path,
        log_dir=log_dir,
        metrics=_read_best_metrics(log_dir),
        error=error,
        diagnosis=diagnosis,
    )


def summarize(results: list[CustomRunResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant in sorted({r.variant for r in results}):
        rows = [r for r in results if r.variant == variant]
        summary: dict[str, Any] = {
            "n_completed": sum(1 for r in rows if r.status == "completed"),
            "n_total": len(rows),
        }
        for metric in ["test_macro_auc", "test_bacc", "test_macro_f1", "test_acc", "val_macro_auc"]:
            values = []
            for row in rows:
                value = row.metrics.get(metric)
                if value is None:
                    continue
                values.append(float(value))
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out[variant] = summary
    return out


def write_report(output_dir: Path, results: list[CustomRunResult], summary: dict[str, dict[str, Any]]) -> Path:
    report = output_dir / "innovation_cv_report.md"
    ranked = sorted(
        summary.items(),
        key=lambda item: item[1].get("test_macro_auc_mean", float("-inf")),
        reverse=True,
    )
    lines = [
        "# Innovation CV Report",
        "",
        "## Mean Metrics",
        "",
        "| Rank | Variant | Completed | Test macro AUC | Test BACC | Test macro F1 | Test ACC | Val macro AUC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (variant, row) in enumerate(ranked, start=1):
        def fmt(name: str) -> str:
            m = row.get(f"{name}_mean")
            s = row.get(f"{name}_std")
            if m is None:
                return ""
            return f"{m:.4f} +/- {s:.4f}"

        lines.append(
            f"| {rank} | {variant} | {row.get('n_completed', 0)}/{row.get('n_total', 0)} | "
            f"{fmt('test_macro_auc')} | {fmt('test_bacc')} | {fmt('test_macro_f1')} | "
            f"{fmt('test_acc')} | {fmt('val_macro_auc')} |"
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


def run_innovation_cv(
    cfg: AutoMilConfig,
    *,
    variants: list[str],
    n_splits: int,
    epochs: int,
    split_plan_path: Path | None = None,
    split_plan_id: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    output_dir = cfg.output_dir / "innovation_cv"
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
        artifacts = prepare_dataset_kfold(
            dataset=dataset,
            task=task,
            output_dir=output_dir / "folds",
            n_splits=n_splits,
        )
    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
    checkpoint.update_metadata(
        command="run-innovation-cv",
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

    results: list[CustomRunResult] = []
    for fold_idx, fold_dir in enumerate(artifacts.fold_dirs):
        dataset_csv = fold_dir / "dataset.csv"
        for variant in variants:
            run_id = f"{variant.lower()}_fold{fold_idx}"
            cached_payload = checkpoint.get_completed_payload(run_id) if resume and not dry_run else None
            if cached_payload is not None:
                result = custom_result_from_payload(cached_payload)
                results.append(result)
                journal.write(
                    "innovation_run_resume",
                    {"fold": fold_idx, "variant": variant, "run_id": run_id, "status": result.status},
                )
                continue
            journal.write("innovation_run_start", {"fold": fold_idx, "variant": variant, "dataset_csv": str(dataset_csv)})
            result = run_custom_variant(
                cfg=cfg,
                dataset_csv=dataset_csv,
                output_dir=output_dir,
                variant=variant,
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
            checkpoint.record_run(result.run_id, "innovation_cv", result.status, payload)
            journal.write("innovation_run_result", payload)

    summary = summarize(results)
    (output_dir / "innovation_cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(output_dir, results, summary)
    checkpoint.update_metadata(report=str(report), summary_json=str(output_dir / "innovation_cv_summary.json"))
    journal.write("innovation_report", {"path": str(report)})
    return report
