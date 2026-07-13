from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .data import prepare_dataset
from .mil_baseline import Recipe, RunResult, run_recipe, run_result_from_payload, run_result_to_payload
from .split_executor import materialize_holdout_from_split_plan, select_split_plan
from .state import ExperimentCheckpoint, ResearchJournal, now_iso


def _now() -> str:
    return now_iso()


def _load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recipe_id(stage: str, model: str, idx: int) -> str:
    return f"{stage}_{idx:02d}_{model.lower()}"


def build_screen_recipes(cfg: AutoMilConfig, max_runs: int | None = None) -> list[Recipe]:
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    if max_runs is not None and max_runs <= 0:
        return []
    models = list(search.get("screen_models", ["MEAN_MIL", "MAX_MIL", "AB_MIL"]))
    if max_runs is not None:
        models = models[:max_runs]
    recipes = []
    for idx, model in enumerate(models):
        recipes.append(
            Recipe(
                recipe_id=_recipe_id("screen", model, idx),
                stage="baseline_screen",
                model_name=model,
                epochs=int(training.get("screening_epochs", 1)),
                lr=float(search.get("learning_rates", [0.0002])[0]),
                dropout=float(search.get("dropouts", [0.1])[0]),
                balanced_sampler=bool(training.get("balanced_sampler", False)),
                notes="Cheap baseline screen adapted from the Camyla stage-1 baseline gate.",
            )
        )
    return recipes


def build_focused_recipes(
    cfg: AutoMilConfig,
    screen_results: list[RunResult],
    max_runs: int | None = None,
) -> list[Recipe]:
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    if max_runs is not None and max_runs <= 0:
        return []
    focused_models = list(search.get("focused_models", ["AB_MIL", "GATE_AB_MIL"]))
    lrs = [float(x) for x in search.get("learning_rates", [0.0002, 0.0001])]
    dropouts = [float(x) for x in search.get("dropouts", [0.1, 0.25])]

    completed = [r for r in screen_results if r.status == "completed"]
    completed.sort(key=lambda r: r.score, reverse=True)
    anchor_model = completed[0].recipe.model_name if completed else focused_models[0]
    ordered_models = [anchor_model] + [m for m in focused_models if m != anchor_model]

    recipes = []
    idx = 0
    for model in ordered_models:
        for lr in lrs:
            for dropout in dropouts:
                recipes.append(
                    Recipe(
                        recipe_id=_recipe_id("focused", model, idx),
                        stage="focused_runs",
                        model_name=model,
                        epochs=int(training.get("focused_epochs", 3)),
                        lr=lr,
                        dropout=dropout,
                        balanced_sampler=True,
                        notes=(
                            "Focused recipe from baseline ranking: tune optimizer/dropout "
                            "and enable class-balanced sampling."
                        ),
                    )
                )
                idx += 1
                if max_runs is not None and len(recipes) >= max_runs:
                    return recipes
    return recipes


def _result_payload(result: RunResult) -> dict[str, Any]:
    return run_result_to_payload(result)


def _write_report(
    output_dir: Path,
    cfg: AutoMilConfig,
    metadata: dict[str, Any],
    screen_results: list[RunResult],
    focused_results: list[RunResult],
) -> Path:
    all_results = screen_results + focused_results
    ranked = sorted(all_results, key=lambda r: r.score, reverse=True)
    report_path = output_dir / "report.md"
    lines = [
        f"# Auto-MIL Report: {cfg.name}",
        "",
        f"Generated: {_now()}",
        "",
        "## Dataset",
        "",
        f"- Source: `{cfg.data_dir}`",
        f"- Task label: `{metadata.get('label_column')}`",
        f"- Classes: `{metadata.get('label_to_id')}`",
        f"- Matched H5 slides: `{metadata.get('num_h5_matched')}`",
        f"- Cases: `{metadata.get('num_cases')}`",
        f"- Feature shape example: `{metadata.get('feature', {}).get('shape')}`",
        "",
        "## Results",
        "",
        "| Rank | Stage | Recipe | Model | Status | Diagnosis | Score | Metrics |",
        "|---:|---|---|---|---|---|---:|---|",
    ]
    for rank, result in enumerate(ranked, start=1):
        metric_path = result.metrics.get("metrics_path", "")
        score = "" if result.score == float("-inf") else f"{result.score:.4f}"
        diagnosis = ""
        if result.diagnosis is not None:
            diagnosis = f"{result.diagnosis.category}: {result.diagnosis.summary}".replace("|", "/")
        lines.append(
            "| "
            f"{rank} | {result.recipe.stage} | `{result.recipe.recipe_id}` | "
            f"{result.recipe.model_name} | {result.status} | {diagnosis} | {score} | `{metric_path}` |"
        )
    lines.extend(["", "## Commands", ""])
    for result in all_results:
        lines.extend(
            [
                f"### {result.recipe.recipe_id}",
                "",
                "```powershell",
                " ".join(f'"{x}"' if " " in x else x for x in result.command),
                "```",
                f"- Config: `{result.config_path}`",
                f"- Stdout: `{result.stdout_path}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Research Hypotheses",
            "",
            "- Compare the best attention-style model against a spatial model using H5 coordinates.",
            "- Add patient-level aggregation if slide multiplicity dominates case labels.",
            "- Run a longer seed sweep once the cheap screen identifies stable candidates.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_autonomous_research(
    cfg: AutoMilConfig,
    *,
    max_screen_runs: int | None = None,
    max_focused_runs: int | None = None,
    split_plan_path: Path | None = None,
    split_plan_id: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")

    task = cfg.task_spec
    dataset = cfg.dataset_spec
    selected_split = None
    if split_plan_path is not None:
        selected_split = select_split_plan(split_plan_path, split_plan_id)
        artifacts = materialize_holdout_from_split_plan(
            dataset=dataset,
            task=task,
            output_dir=output_dir,
            split_plan_path=split_plan_path,
            plan_id=split_plan_id,
        )
    else:
        artifacts = prepare_dataset(
            dataset=dataset,
            task=task,
            output_dir=output_dir,
        )
    metadata = _load_metadata(artifacts.metadata_json)
    checkpoint.update_metadata(
        command="run",
        config=str(cfg.path),
        dataset_csv=str(artifacts.dataset_csv),
        metadata_json=str(artifacts.metadata_json),
        dry_run=dry_run,
        split_plan=str(split_plan_path) if split_plan_path else None,
        split_plan_id=selected_split.plan_id if selected_split else None,
    )
    journal.write(
        "dataset_audit",
        {
            "dataset_csv": str(artifacts.dataset_csv),
            "h5_paths_csv": str(artifacts.h5_paths_csv),
            "metadata_json": str(artifacts.metadata_json),
            "metadata": metadata,
        },
    )

    training = cfg.raw.get("training", {})
    in_dim_cfg = training.get("in_dim", "auto")
    in_dim = int(metadata["feature"]["in_dim"] if in_dim_cfg == "auto" else in_dim_cfg)
    common = {
        "python": cfg.python,
        "mil_baseline_dir": cfg.mil_baseline_dir,
        "dataset_name": cfg.name,
        "dataset_csv_path": artifacts.dataset_csv,
        "output_dir": output_dir,
        "num_classes": int(metadata["num_classes"]),
        "in_dim": in_dim,
        "device": int(training.get("device", 0)),
        "num_workers": int(training.get("num_workers", 0)),
        "best_metric": str(training.get("best_metric", "macro_auc")),
        "dry_run": dry_run,
    }

    screen_results = []
    for recipe in build_screen_recipes(cfg, max_screen_runs):
        cached_payload = checkpoint.get_completed_payload(recipe.recipe_id) if resume and not dry_run else None
        if cached_payload is not None:
            result = run_result_from_payload(cached_payload)
            screen_results.append(result)
            journal.write("baseline_screen_resume", {"recipe": asdict(recipe), "status": result.status})
            continue
        journal.write("baseline_screen_start", {"recipe": asdict(recipe)})
        result = run_recipe(recipe, **common)
        screen_results.append(result)
        payload = _result_payload(result)
        checkpoint.record_run(recipe.recipe_id, recipe.stage, result.status, payload)
        journal.write("baseline_screen_result", payload)

    focused_results = []
    for recipe in build_focused_recipes(cfg, screen_results, max_focused_runs):
        cached_payload = checkpoint.get_completed_payload(recipe.recipe_id) if resume and not dry_run else None
        if cached_payload is not None:
            result = run_result_from_payload(cached_payload)
            focused_results.append(result)
            journal.write("focused_run_resume", {"recipe": asdict(recipe), "status": result.status})
            continue
        journal.write("focused_run_start", {"recipe": asdict(recipe)})
        result = run_recipe(recipe, **common)
        focused_results.append(result)
        payload = _result_payload(result)
        checkpoint.record_run(recipe.recipe_id, recipe.stage, result.status, payload)
        journal.write("focused_run_result", payload)

    report = _write_report(output_dir, cfg, metadata, screen_results, focused_results)
    checkpoint.update_metadata(report=str(report))
    journal.write("report", {"path": str(report)})
    return report
