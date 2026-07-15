from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import run_ablation_cv
from .autonomous_window import AutonomousContract, run_autonomous_window
from .config import load_config
from .baseline_registry import available_models, resolve_mil_baseline_dir
from .baseline_families import baseline_plan_to_payload, build_baseline_plan, write_baseline_plan
from .data import prepare_cptac_brca, prepare_dataset
from .cv import run_case_level_cv
from .experiment_tree import run_qwbe_lite
from .failure_policy import action_to_payload, decide_failure_action
from .figure_report import build_figure_report
from .innovation_cv import run_innovation_cv
from .log_analyzer import diagnose_log_file, diagnosis_to_payload
from .manuscript_package import available_manuscript_profiles, package_manuscript
from .manuscript_writer import write_manuscript_draft
from .prediction_aggregator import aggregate_predictions
from .proposal_generator import propose_nodes
from .research import run_autonomous_research
from .result_collector import collect_results, write_result_bundle
from .split_planner import plan_splits, split_plan_to_payload, write_split_plan
from .specs import describe_capabilities, specs_to_payload
from .stats_analysis import build_stats_report, stats_report_to_payload, write_stats_report
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


def cmd_prepare_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifacts = prepare_dataset(
        dataset=cfg.dataset_spec,
        task=cfg.task_spec,
        output_dir=Path(args.output_dir) if args.output_dir else cfg.output_dir / "dataset",
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
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_inspect_spec(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    task = cfg.task_spec
    dataset = cfg.dataset_spec
    payload = specs_to_payload(task, dataset)
    payload["capabilities"] = describe_capabilities(task, dataset)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return
    print(f"experiment={cfg.name}")
    print(f"task_kind={task.kind}")
    print(f"outcome_column={task.outcome_column}")
    print(f"dataset={dataset.name}")
    print(f"bag_level={dataset.bag_level}")
    print(f"data_dir={dataset.data_dir}")
    print(f"labels_csv={dataset.labels_csv}")
    print(f"labels_sheet={dataset.labels_sheet}")
    print(f"case_id_column={dataset.case_id_column}")
    print(f"slide_path_column={dataset.slide_path_column}")
    print(f"feature_format={dataset.feature.format}")
    print(f"feature_key={dataset.feature.feature_key}")
    print(f"coords_key={dataset.feature.coords_key}")
    print(f"feature_glob={dataset.feature.feature_glob}")
    print(f"can_prepare_mil_baseline={str(payload['capabilities']['can_prepare_mil_baseline']).lower()}")
    if payload["capabilities"]["blocked_reason"]:
        print(f"blocked_reason={payload['capabilities']['blocked_reason']}")


def cmd_plan_split(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    bundle = plan_splits(cfg.dataset_spec, cfg.task_spec)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir / "split_plan"
    json_path, md_path = write_split_plan(bundle, output_dir)
    if args.json:
        print(json.dumps(split_plan_to_payload(bundle), indent=2, ensure_ascii=True))
        return
    recommended = [plan for plan in bundle.plans if plan.recommended]
    print(f"split_plan_json={json_path}")
    print(f"split_plan_md={md_path}")
    print(f"cases={bundle.profile.num_cases}")
    print(f"slides={bundle.profile.num_slides}")
    print(f"classes={bundle.profile.class_counts}")
    print("recommended_plans=" + ",".join(plan.plan_id for plan in recommended))
    for plan in recommended:
        print(f"  {plan.plan_id}: {plan.strategy} ({plan.rationale})")


def cmd_plan_baselines(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    models = args.models.split(",") if args.models else None
    plan = build_baseline_plan(
        dataset=cfg.dataset_spec,
        task=cfg.task_spec,
        mil_baseline_dir=cfg.mil_baseline_dir,
        models=models,
    )
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir / "baseline_plan"
    json_path, md_path = write_baseline_plan(plan, output_dir)
    if args.json:
        print(json.dumps(baseline_plan_to_payload(plan), indent=2, ensure_ascii=True))
        return
    print(f"baseline_plan_json={json_path}")
    print(f"baseline_plan_md={md_path}")
    print(f"has_real_coords={str(plan.has_real_coords).lower()}")
    print("recommended_screen=" + ",".join(plan.recommended_screen))
    for item in plan.assessments:
        if item.recommended_for_screen or item.warnings:
            print(
                f"  {item.model_name}: compatible={str(item.compatible).lower()} "
                f"tier={item.tier} coords={item.coordinate_policy} warnings={'; '.join(item.warnings)}"
            )


def cmd_run_cv(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    models = args.models.split(",") if args.models else None
    report = run_case_level_cv(
        cfg,
        models=models,
        n_splits=args.n_splits,
        epochs=args.epochs,
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
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
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_run_ablation_cv(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    variants = args.variants.split(",") if args.variants else None
    report = run_ablation_cv(
        cfg,
        variants=variants,
        n_splits=args.n_splits,
        epochs=args.epochs,
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
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
        max_failure_retry_depth=args.max_failure_retry_depth,
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(f"report={report}")


def cmd_run_autonomous_window(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    contract = AutonomousContract(
        max_minutes=args.max_minutes,
        max_runs=args.max_runs,
        target_metric=args.target_metric,
        target_value=args.target_value,
        per_run_timeout_seconds=args.timeout_seconds,
        split_plan_path=Path(args.split_plan) if args.split_plan else None,
        split_plan_id=args.plan_id,
        max_screen_models=args.max_screen_models,
        max_children_per_parent=args.max_children_per_parent,
        max_failure_retry_depth=args.max_failure_retry_depth,
        enable_proposals=not args.disable_proposals,
        baseline_plan_path=Path(args.baseline_plan) if args.baseline_plan else None,
        max_proposals_per_round=args.max_proposals_per_round,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    summary = run_autonomous_window(cfg, contract)
    print(f"summary={summary}")


def cmd_propose_nodes(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = propose_nodes(
        cfg,
        tree_path=Path(args.tree_path) if args.tree_path else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        baseline_plan_path=Path(args.baseline_plan) if args.baseline_plan else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_proposals=args.max_proposals,
        apply=not args.no_apply,
    )
    print(f"proposal_report={report}")


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


def cmd_failure_action(args: argparse.Namespace) -> None:
    diagnosis = diagnose_log_file(Path(args.path))
    action = decide_failure_action(diagnosis)
    payload = {
        "path": str(Path(args.path)),
        "diagnosis": diagnosis_to_payload(diagnosis),
        "failure_action": action_to_payload(action),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return
    print(f"path={payload['path']}")
    print(f"diagnosis={diagnosis.category}")
    print(f"severity={diagnosis.severity}")
    print(f"policy_action={action.action}")
    print(f"retryable={str(action.retryable).lower()}")
    print(f"summary={action.summary}")
    if action.config_overrides:
        print("config_overrides=" + json.dumps(action.config_overrides, sort_keys=True))


def cmd_analyze_stats(args: argparse.Namespace) -> None:
    report = build_stats_report(args.checkpoint, args.metric, args.baseline)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "stats"
    json_path, md_path = write_stats_report(report, output_dir)
    if args.json:
        print(json.dumps(stats_report_to_payload(report), indent=2, ensure_ascii=True))
        return
    print(f"stats_report_json={json_path}")
    print(f"stats_report_md={md_path}")
    print(f"metric={report.metric}")
    for row in report.summaries:
        print(f"  {row.model_name}: n={row.n} mean={row.mean} ci95=({row.ci95_low}, {row.ci95_high})")
    for warning in report.warnings:
        print(f"warning={warning}")


def cmd_collect_results(args: argparse.Namespace) -> None:
    root = Path(args.root)
    checkpoints = [Path(path) for path in args.checkpoint] if args.checkpoint else None
    bundle = collect_results(
        root,
        checkpoint_paths=checkpoints,
        primary_metric=args.primary_metric,
        metrics=args.metrics.split(",") if args.metrics else None,
        include_failed=not args.completed_only,
    )
    output_dir = Path(args.output_dir) if args.output_dir else root / "results"
    json_path, runs_csv, summary_csv, report_md = write_result_bundle(bundle, output_dir)
    if args.json:
        from dataclasses import asdict

        from .state import json_ready

        print(json.dumps(json_ready(asdict(bundle)), indent=2, ensure_ascii=True))
        return
    print(f"results_index={json_path}")
    print(f"runs_csv={runs_csv}")
    print(f"model_summary_csv={summary_csv}")
    print(f"manuscript_results={report_md}")
    print(f"checkpoints={len(bundle.checkpoints)}")
    print(f"runs={len(bundle.runs)}")
    for warning in bundle.warnings:
        print(f"warning={warning}")


def cmd_aggregate_predictions(args: argparse.Namespace) -> None:
    prediction_paths = [Path(path) for path in args.prediction] if args.prediction else None
    bundle, slide_csv, case_csv, metrics_json, report_md = aggregate_predictions(
        args.root,
        prediction_paths=prediction_paths,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        case_id_regex=args.case_id_regex,
        aggregation=args.aggregation,
    )
    if args.json:
        from dataclasses import asdict

        from .state import json_ready

        print(json.dumps(json_ready(asdict(bundle)), indent=2, ensure_ascii=True))
        return
    print(f"slide_predictions={slide_csv}")
    print(f"case_predictions={case_csv}")
    print(f"case_metrics={metrics_json}")
    print(f"prediction_report={report_md}")
    print(f"prediction_files={len(bundle.prediction_files)}")
    print(f"slide_predictions_n={bundle.num_slide_predictions}")
    print(f"case_predictions_n={bundle.num_case_predictions}")
    for warning in bundle.warnings:
        print(f"warning={warning}")


def cmd_build_figures(args: argparse.Namespace) -> None:
    case_predictions = Path(args.case_predictions) if args.case_predictions else Path(args.root) / "prediction_aggregation" / "case_predictions.csv"
    class_names = args.class_names.split(",") if args.class_names else None
    report, json_path, md_path = build_figure_report(
        case_predictions,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        class_names=class_names,
        positive_class=args.positive_class,
    )
    if args.json:
        from dataclasses import asdict

        from .state import json_ready

        print(json.dumps(json_ready(asdict(report)), indent=2, ensure_ascii=True))
        return
    print(f"figure_report_json={json_path}")
    print(f"figure_report_md={md_path}")
    print(f"artifacts={len(report.artifacts)}")
    for item in report.artifacts:
        print(f"  {item.name}={item.path}")
    for warning in report.warnings:
        print(f"warning={warning}")


def cmd_write_manuscript(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else None
    draft, json_path, md_path = write_manuscript_draft(
        args.root,
        cfg=cfg,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        title=args.title,
        primary_metric=args.primary_metric,
    )
    if args.json:
        from dataclasses import asdict

        from .state import json_ready

        print(json.dumps(json_ready(asdict(draft)), indent=2, ensure_ascii=True))
        return
    print(f"manuscript_evidence_json={json_path}")
    print(f"manuscript_draft={md_path}")
    for warning in draft.warnings:
        print(f"warning={warning}")


def cmd_package_manuscript(args: argparse.Namespace) -> None:
    package, manifest_path = package_manuscript(
        args.root,
        draft_path=Path(args.draft) if args.draft else None,
        evidence_path=Path(args.evidence) if args.evidence else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        profile=args.profile,
        target_journal=args.target_journal,
    )
    if args.json:
        from dataclasses import asdict

        from .state import json_ready

        print(json.dumps(json_ready(asdict(package)), indent=2, ensure_ascii=True))
        return
    print(f"manuscript_package={manifest_path}")
    for name, path in package.outputs.items():
        if name != "manifest":
            print(f"{name}={path}")
    for warning in package.warnings:
        print(f"warning={warning}")


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

    prep_data = sub.add_parser("prepare-data", help="Prepare a generic configured dataset for MIL_BASELINE")
    prep_data.add_argument("--config", required=True)
    prep_data.add_argument("--output-dir", default=None, help="Directory for dataset.csv/h5_paths.csv/metadata.json")
    prep_data.set_defaults(func=cmd_prepare_data)

    inspect = sub.add_parser("inspect-spec", help="Show normalized TaskSpec/DatasetSpec for a config")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--json", action="store_true", help="Print JSON")
    inspect.set_defaults(func=cmd_inspect_spec)

    split = sub.add_parser("plan-split", help="Inspect data and propose manuscript-grade split plans")
    split.add_argument("--config", required=True)
    split.add_argument("--output-dir", default=None, help="Directory for split_plan.json/md")
    split.add_argument("--json", action="store_true", help="Print JSON payload")
    split.set_defaults(func=cmd_plan_split)

    baseline_plan = sub.add_parser("plan-baselines", help="Assess and recommend MIL baseline families")
    baseline_plan.add_argument("--config", required=True)
    baseline_plan.add_argument("--models", default=None, help="Comma-separated model list to assess")
    baseline_plan.add_argument("--output-dir", default=None, help="Directory for baseline_plan.json/md")
    baseline_plan.add_argument("--json", action="store_true", help="Print JSON payload")
    baseline_plan.set_defaults(func=cmd_plan_baselines)

    run = sub.add_parser("run", help="Run the staged autonomous research loop")
    run.add_argument("--config", required=True)
    run.add_argument("--max-screen-runs", type=int, default=None)
    run.add_argument("--max-focused-runs", type=int, default=None)
    run.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for holdout/external/center plans")
    run.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    run.set_defaults(func=cmd_run)

    cv = sub.add_parser("run-cv", help="Run case-level k-fold CV baselines")
    cv.add_argument("--config", required=True)
    cv.add_argument("--n-splits", type=int, default=5)
    cv.add_argument("--epochs", type=int, default=None)
    cv.add_argument("--models", default=None, help="Comma-separated model list")
    cv.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for CV plans")
    cv.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    cv.add_argument("--dry-run", action="store_true")
    cv.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    cv.set_defaults(func=cmd_run_cv)

    innovation = sub.add_parser("run-innovation-cv", help="Run custom AB_MIL innovation k-fold CV")
    innovation.add_argument("--config", required=True)
    innovation.add_argument("--n-splits", type=int, default=5)
    innovation.add_argument("--epochs", type=int, default=3)
    innovation.add_argument("--variants", default="AB_MIL_FOCAL,AB_MIL_FOCAL_PROTO")
    innovation.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for CV plans")
    innovation.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    innovation.add_argument("--dry-run", action="store_true")
    innovation.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    innovation.set_defaults(func=cmd_run_innovation_cv)

    ablation = sub.add_parser("run-ablation-cv", help="Run AB_MIL innovation ablation k-fold CV")
    ablation.add_argument("--config", required=True)
    ablation.add_argument("--n-splits", type=int, default=5)
    ablation.add_argument("--epochs", type=int, default=3)
    ablation.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants; default is AB_MIL_CE,AB_MIL_FOCAL,AB_MIL_PROTO,AB_MIL_FOCAL_PROTO",
    )
    ablation.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for CV plans")
    ablation.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    ablation.add_argument("--dry-run", action="store_true")
    ablation.add_argument("--resume", action="store_true", help="Reuse completed runs from checkpoint.json")
    ablation.set_defaults(func=cmd_run_ablation_cv)

    tree = sub.add_parser("run-tree", help="Run QWBE-lite experiment-tree search")
    tree.add_argument("--config", required=True)
    tree.add_argument("--max-runs", type=int, default=6)
    tree.add_argument("--max-screen-models", type=int, default=None)
    tree.add_argument("--max-children-per-parent", type=int, default=4)
    tree.add_argument("--max-failure-retry-depth", type=int, default=1)
    tree.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for holdout/external/center plans")
    tree.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    tree.add_argument("--timeout-seconds", type=int, default=None, help="Per-node training timeout")
    tree.add_argument("--dry-run", action="store_true")
    tree.add_argument("--resume", action="store_true", help="Reuse completed nodes from checkpoint.json")
    tree.set_defaults(func=cmd_run_tree)

    window = sub.add_parser("run-autonomous-window", help="Run a time-boxed autonomous experiment loop")
    window.add_argument("--config", required=True)
    window.add_argument("--max-minutes", type=float, required=True)
    window.add_argument("--max-runs", type=int, required=True)
    window.add_argument("--target-metric", default="test_macro_auc")
    window.add_argument("--target-value", type=float, default=None)
    window.add_argument("--timeout-seconds", type=int, default=None, help="Per-node training timeout")
    window.add_argument("--split-plan", default=None, help="Confirmed split_plan.json for holdout/external/center plans")
    window.add_argument("--plan-id", default=None, help="Plan id inside split_plan.json")
    window.add_argument("--max-screen-models", type=int, default=None)
    window.add_argument("--max-children-per-parent", type=int, default=4)
    window.add_argument("--max-failure-retry-depth", type=int, default=1)
    window.add_argument("--baseline-plan", default=None, help="baseline_plan.json used for proposal generation")
    window.add_argument("--max-proposals-per-round", type=int, default=4)
    window.add_argument("--disable-proposals", action="store_true", help="Do not generate new tree nodes when pending nodes are exhausted")
    window.add_argument("--dry-run", action="store_true")
    window.add_argument("--resume", action="store_true", help="Reuse completed nodes from checkpoint.json")
    window.set_defaults(func=cmd_run_autonomous_window)

    propose = sub.add_parser("propose-nodes", help="Generate candidate ExperimentTree nodes from current evidence")
    propose.add_argument("--config", required=True)
    propose.add_argument("--tree-path", default=None, help="experiment_tree.json path")
    propose.add_argument("--checkpoint", default=None, help="checkpoint.json path")
    propose.add_argument("--baseline-plan", default=None, help="baseline_plan.json path")
    propose.add_argument("--output-dir", default=None, help="Directory for proposal_report.json/md")
    propose.add_argument("--max-proposals", type=int, default=6)
    propose.add_argument("--no-apply", action="store_true", help="Preview proposals without adding nodes")
    propose.set_defaults(func=cmd_propose_nodes)

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

    failure = sub.add_parser("failure-action", help="Classify a log and print the retry/escalation policy")
    failure.add_argument("--path", required=True)
    failure.add_argument("--json", action="store_true", help="Print JSON record")
    failure.set_defaults(func=cmd_failure_action)

    stats = sub.add_parser("analyze-stats", help="Compute fold/seed statistical summaries from a checkpoint")
    stats.add_argument("--checkpoint", required=True, help="checkpoint.json path")
    stats.add_argument("--metric", default="test_macro_auc")
    stats.add_argument("--baseline", default=None, help="Baseline model for paired comparisons")
    stats.add_argument("--output-dir", default=None, help="Directory for stats_report.json/md")
    stats.add_argument("--json", action="store_true", help="Print JSON payload")
    stats.set_defaults(func=cmd_analyze_stats)

    collect = sub.add_parser("collect-results", help="Collect checkpoints into manuscript-ready result tables")
    collect.add_argument("--root", required=True, help="Run root containing checkpoint.json files")
    collect.add_argument("--checkpoint", action="append", default=None, help="Specific checkpoint.json path; repeatable")
    collect.add_argument("--primary-metric", default="test_macro_auc")
    collect.add_argument("--metrics", default=None, help="Comma-separated metrics to collect")
    collect.add_argument("--output-dir", default=None, help="Directory for results_index/runs/model_summary/report")
    collect.add_argument("--completed-only", action="store_true", help="Exclude failed and dry-run records")
    collect.add_argument("--json", action="store_true", help="Print JSON payload")
    collect.set_defaults(func=cmd_collect_results)

    aggregate = sub.add_parser("aggregate-predictions", help="Aggregate slide-level prediction CSVs to case level")
    aggregate.add_argument("--root", required=True, help="Run root or prediction CSV path")
    aggregate.add_argument("--prediction", action="append", default=None, help="Specific prediction CSV path; repeatable")
    aggregate.add_argument("--output-dir", default=None, help="Directory for slide/case prediction artifacts")
    aggregate.add_argument("--case-id-regex", default=r"^(?P<case>[A-Za-z0-9]+)-")
    aggregate.add_argument("--aggregation", choices=["mean", "median", "max"], default="mean")
    aggregate.add_argument("--json", action="store_true", help="Print JSON payload")
    aggregate.set_defaults(func=cmd_aggregate_predictions)

    figures = sub.add_parser("build-figures", help="Build manuscript figures from case-level predictions")
    figures.add_argument("--root", default=".", help="Run root used when --case-predictions is omitted")
    figures.add_argument("--case-predictions", default=None, help="case_predictions.csv path")
    figures.add_argument("--output-dir", default=None, help="Directory for figure PNGs and tables")
    figures.add_argument("--class-names", default=None, help="Comma-separated class names indexed by class id")
    figures.add_argument("--positive-class", type=int, default=1, help="Positive class for binary ROC/calibration")
    figures.add_argument("--json", action="store_true", help="Print JSON payload")
    figures.set_defaults(func=cmd_build_figures)

    manuscript = sub.add_parser("write-manuscript", help="Draft Methods and Results from collected Auto-MIL artifacts")
    manuscript.add_argument("--root", required=True, help="Run root containing result artifacts")
    manuscript.add_argument("--config", default=None, help="Optional Auto-MIL config for task/dataset wording")
    manuscript.add_argument("--output-dir", default=None, help="Directory for manuscript_draft.md")
    manuscript.add_argument("--title", default=None)
    manuscript.add_argument("--primary-metric", default="test_macro_auc")
    manuscript.add_argument("--json", action="store_true", help="Print JSON payload")
    manuscript.set_defaults(func=cmd_write_manuscript)

    package = sub.add_parser("package-manuscript", help="Package a manuscript draft for polishing and submission checks")
    package.add_argument("--root", required=True, help="Run root containing manuscript artifacts")
    package.add_argument("--draft", default=None, help="Optional manuscript_draft.md path")
    package.add_argument("--evidence", default=None, help="Optional manuscript_evidence.json path")
    package.add_argument("--output-dir", default=None, help="Directory for package artifacts")
    package.add_argument("--profile", choices=available_manuscript_profiles(), default="generic-pathology-ai")
    package.add_argument("--target-journal", default=None, help="Free-text target journal/style label")
    package.add_argument("--json", action="store_true", help="Print JSON payload")
    package.set_defaults(func=cmd_package_manuscript)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
