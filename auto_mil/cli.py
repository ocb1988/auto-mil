from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .baseline_registry import available_models, resolve_mil_baseline_dir
from .data import prepare_cptac_brca
from .cv import run_case_level_cv
from .experiment_tree import run_qwbe_lite
from .innovation_cv import run_innovation_cv
from .log_analyzer import diagnose_log_file, diagnosis_to_payload
from .research import run_autonomous_research
from .state import ExperimentCheckpoint


def cmd_prepare_cptac(args: argparse.Namespace) -> None:
    artifacts = prepare_cptac_brca(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv or Path(args.data_dir) / "CPTAC-BRCA_clinical_labels.csv",
        output_dir=args.output_dir,
        label_column=args.label_column,
        min_class_count=args.min_class_count,
        seed=args.seed,
    )
    print(f"dataset_csv={artifacts.dataset_csv}")
    print(f"h5_paths_csv={artifacts.h5_paths_csv}")
    print(f"metadata_json={artifacts.metadata_json}")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = run_autonomous_research(
        cfg,
        max_screen_runs=args.max_screen_runs,
        max_focused_runs=args.max_focused_runs,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_run_cv(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    models = args.models.split(",") if args.models else None
    report = run_case_level_cv(
        cfg,
        models=models,
        n_splits=args.n_splits,
        epochs=args.epochs,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_run_innovation_cv(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    variants = args.variants.split(",")
    report = run_innovation_cv(
        cfg,
        variants=variants,
        n_splits=args.n_splits,
        epochs=args.epochs,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_run_tree(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = run_qwbe_lite(
        cfg,
        max_runs=args.max_runs,
        max_screen_models=args.max_screen_models,
        max_children_per_parent=args.max_children_per_parent,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_list_baselines(args: argparse.Namespace) -> None:
    root = resolve_mil_baseline_dir(args.mil_baseline_dir)
    registry = available_models(root)
    print(f"mil_baseline_dir={root}")
    print("model,trainable,config,module,process")
    for name, info in registry.items():
        if args.trainable_only and not info.is_trainable:
            continue
        print(
            f"{name},{str(info.is_trainable).lower()},"
            f"{str(info.has_config).lower()},"
            f"{str(info.has_module).lower()},"
            f"{str(info.has_process).lower()}"
        )


def cmd_checkpoint_summary(args: argparse.Namespace) -> None:
    path = Path(args.path)
    checkpoint_path = path / "checkpoint.json" if path.is_dir() else path
    checkpoint = ExperimentCheckpoint(checkpoint_path)
    runs = checkpoint.state.get("runs", {})
    by_status: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for run in runs.values():
        status = str(run.get("status", "unknown"))
        stage = str(run.get("stage", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
    print(f"checkpoint={checkpoint_path}")
    print(f"updated_at={checkpoint.state.get('updated_at')}")
    print(f"runs={len(runs)}")
    print("status_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(by_status.items())))
    print("stage_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(by_stage.items())))


def cmd_analyze_log(args: argparse.Namespace) -> None:
    path = Path(args.path)
    files = sorted(path.rglob("*.log")) if path.is_dir() else [path]
    records = []
    for file_path in files:
        diagnosis = diagnose_log_file(file_path)
        payload = {"path": str(file_path), "diagnosis": diagnosis_to_payload(diagnosis)}
        records.append(payload)
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=True))
        return
    for record in records:
        diag = record["diagnosis"] or {}
        print(f"{record['path']}")
        print(f"  category: {diag.get('category')}")
        print(f"  severity: {diag.get('severity')}")
        print(f"  summary: {diag.get('summary')}")
        print(f"  action: {diag.get('recommended_action')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous MIL research runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-cptac", help="Prepare CPTAC-BRCA H5 data for MIL_BASELINE")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--labels-csv", default=None)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--label-column", default="PAM50")
    prep.add_argument("--min-class-count", type=int, default=2)
    prep.add_argument("--seed", type=int, default=2024)
    prep.set_defaults(func=cmd_prepare_cptac)

    run = sub.add_parser("run", help="Run the staged autonomous research loop")
    run.add_argument("--config", required=True)
    run.add_argument("--max-screen-runs", type=int, default=None)
    run.add_argument("--max-focused-runs", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    run.set_defaults(func=cmd_run)

    cv = sub.add_parser("run-cv", help="Run case-level k-fold CV baselines")
    cv.add_argument("--config", required=True)
    cv.add_argument("--n-splits", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=None)
    cv.add_argument("--models", default=None, help="Comma-separated model list")
    cv.add_argument("--dry-run", action="store_true")
    cv.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    cv.set_defaults(func=cmd_run_cv)

    innovation = sub.add_parser("run-innovation-cv", help="Run custom AB_MIL innovation k-fold CV")
    innovation.add_argument("--config", required=True)
    innovation.add_argument("--n-splits", type=int, default=5)
    innovation.add_argument("--epochs", type=int, default=3)
    innovation.add_argument("--variants", default="AB_MIL_FOCAL,AB_MIL_FOCAL_PROTO")
    innovation.add_argument("--dry-run", action="store_true")
    innovation.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    innovation.set_defaults(func=cmd_run_innovation_cv)

    tree = sub.add_parser("run-tree", help="Run QWBE-lite experiment-tree search")
    tree.add_argument("--config", required=True)
    tree.add_argument("--max-runs", type=int, default=6)
    tree.add_argument("--max-screen-models", type=int, default=None)
    tree.add_argument("--max-children-per-parent", type=int, default=4)
    tree.add_argument("--dry-run", action="store_true")
    tree.add_argument("--resume", action="store_true", help="Reuse completed nodes from checkpoint.json")
    tree.set_defaults(func=cmd_run_tree)

    baselines = sub.add_parser("list-baselines", help="List bundled or configured MIL_BASELINE models")
    baselines.add_argument("--mil-baseline-dir", default="bundled")
    baselines.add_argument("--trainable-only", action="store_true")
    baselines.set_defaults(func=cmd_list_baselines)

    checkpoint = sub.add_parser("checkpoint-summary", help="Summarize an Auto-MIL checkpoint")
    checkpoint.add_argument("--path", required=True, help="Run directory or checkpoint.json path")
    checkpoint.set_defaults(func=cmd_checkpoint_summary)

    analyze = sub.add_parser("analyze-log", help="Classify a log file or directory of .log files")
    analyze.add_argument("--path", required=True)
    analyze.add_argument("--json", action="store_true", help="Print JSON records")
    analyze.set_defaults(func=cmd_analyze_log)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
