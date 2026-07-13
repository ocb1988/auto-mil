from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .experiment_tree import ExperimentTree, run_qwbe_lite
from .state import ResearchJournal, json_ready, now_iso


@dataclass(frozen=True)
class AutonomousContract:
    max_minutes: float
    max_runs: int
    target_metric: str
    target_value: float | None = None
    per_run_timeout_seconds: int | None = None
    split_plan_path: Path | None = None
    split_plan_id: str | None = None
    max_screen_models: int | None = None
    max_children_per_parent: int = 4
    max_failure_retry_depth: int = 1
    dry_run: bool = False
    resume: bool = True


@dataclass(frozen=True)
class AutonomousRound:
    round_index: int
    started_at: str
    finished_at: str
    report_path: str
    best_metric_value: float | None
    best_run_id: str | None
    best_model: str | None
    stop_reason: str | None = None


@dataclass(frozen=True)
class AutonomousSummary:
    config: str
    output_dir: str
    started_at: str
    finished_at: str
    contract: dict[str, Any]
    rounds: list[AutonomousRound]
    stop_reason: str
    best_metric_value: float | None
    best_run_id: str | None
    best_model: str | None
    tree_json: str | None = None
    warnings: list[str] = field(default_factory=list)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_metric(checkpoint_path: Path, metric: str) -> tuple[float | None, str | None, str | None]:
    checkpoint = _load_checkpoint(checkpoint_path)
    best_value: float | None = None
    best_run_id: str | None = None
    best_model: str | None = None
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
        if best_value is None or value > best_value:
            best_value = value
            best_run_id = str(run_id)
            recipe = payload.get("recipe", {})
            best_model = str(recipe.get("model_name", "unknown"))
    return best_value, best_run_id, best_model


def _tree_pending_count(tree_path: Path) -> int:
    if not tree_path.exists():
        return 0
    tree = ExperimentTree(tree_path)
    return len(tree.pending_nodes())


def _checkpoint_run_count(checkpoint_path: Path) -> int:
    return len(_load_checkpoint(checkpoint_path).get("runs", {}))


def _write_summary(summary: AutonomousSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "autonomous_summary.json"
    md_path = output_dir / "autonomous_summary.md"
    payload = json_ready(asdict(summary))
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _render_markdown(summary: AutonomousSummary) -> str:
    lines = [
        "# Autonomous Window Summary",
        "",
        f"- Started: `{summary.started_at}`",
        f"- Finished: `{summary.finished_at}`",
        f"- Stop reason: `{summary.stop_reason}`",
        f"- Best run: `{summary.best_run_id}`",
        f"- Best model: `{summary.best_model}`",
        f"- Best metric value: `{summary.best_metric_value}`",
        f"- Tree: `{summary.tree_json}`",
        "",
        "## Contract",
        "",
        "```json",
        json.dumps(summary.contract, indent=2, ensure_ascii=True),
        "```",
        "",
        "## Rounds",
        "",
        "| Round | Started | Finished | Best | Best run | Stop reason | Report |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for item in summary.rounds:
        value = "" if item.best_metric_value is None else f"{item.best_metric_value:.4f}"
        lines.append(
            f"| {item.round_index} | {item.started_at} | {item.finished_at} | {value} | "
            f"`{item.best_run_id or ''}` | {item.stop_reason or ''} | `{item.report_path}` |"
        )
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary.warnings])
    return "\n".join(lines)


def run_autonomous_window(cfg: AutoMilConfig, contract: AutonomousContract) -> Path:
    output_dir = cfg.output_dir / "autonomous_window"
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = ResearchJournal(output_dir / "autonomous_journal.jsonl")
    started_at = now_iso()
    deadline = time.monotonic() + max(0.0, contract.max_minutes) * 60.0
    rounds: list[AutonomousRound] = []
    warnings: list[str] = []
    stop_reason = "max_runs_reached"
    tree_output_dir = cfg.output_dir / "experiment_tree"
    checkpoint_path = tree_output_dir / "checkpoint.json"
    tree_path = tree_output_dir / "experiment_tree.json"

    journal.write("autonomous_window_start", {"config": str(cfg.path), "contract": asdict(contract)})

    for round_index in range(1, contract.max_runs + 1):
        if time.monotonic() >= deadline:
            stop_reason = "time_budget_exhausted"
            break
        round_started = now_iso()
        best_before, _run_before, _model_before = _best_metric(checkpoint_path, contract.target_metric)
        journal.write(
            "autonomous_round_start",
            {
                "round": round_index,
                "best_before": best_before,
                "pending_nodes": _tree_pending_count(tree_path),
            },
        )
        run_count_before = _checkpoint_run_count(checkpoint_path)
        report_path = run_qwbe_lite(
            cfg,
            max_runs=1,
            max_screen_models=contract.max_screen_models,
            max_children_per_parent=contract.max_children_per_parent,
            max_failure_retry_depth=contract.max_failure_retry_depth,
            split_plan_path=contract.split_plan_path,
            split_plan_id=contract.split_plan_id,
            timeout_seconds=contract.per_run_timeout_seconds,
            dry_run=contract.dry_run,
            resume=contract.resume,
        )
        run_count_after = _checkpoint_run_count(checkpoint_path)
        best_value, best_run_id, best_model = _best_metric(checkpoint_path, contract.target_metric)
        round_stop = None
        if run_count_after == run_count_before and _tree_pending_count(tree_path) == 0:
            round_stop = "no_pending_nodes"
            stop_reason = "no_pending_nodes"
        if contract.target_value is not None and best_value is not None and best_value >= contract.target_value:
            round_stop = "target_reached"
            stop_reason = "target_reached"
        rounds.append(
            AutonomousRound(
                round_index=round_index,
                started_at=round_started,
                finished_at=now_iso(),
                report_path=str(report_path),
                best_metric_value=best_value,
                best_run_id=best_run_id,
                best_model=best_model,
                stop_reason=round_stop,
            )
        )
        journal.write(
            "autonomous_round_result",
            {
                "round": round_index,
                "report": str(report_path),
                "best_value": best_value,
                "best_run_id": best_run_id,
                "best_model": best_model,
                "stop_reason": round_stop,
            },
        )
        if round_stop:
            break
    else:
        stop_reason = "max_runs_reached"

    if contract.dry_run:
        warnings.append("This was a dry run; no training metrics are expected.")
    best_value, best_run_id, best_model = _best_metric(checkpoint_path, contract.target_metric)
    summary = AutonomousSummary(
        config=str(cfg.path),
        output_dir=str(output_dir),
        started_at=started_at,
        finished_at=now_iso(),
        contract=json_ready(asdict(contract)),
        rounds=rounds,
        stop_reason=stop_reason,
        best_metric_value=best_value,
        best_run_id=best_run_id,
        best_model=best_model,
        tree_json=str(tree_path) if tree_path.exists() else None,
        warnings=warnings,
    )
    _json_path, md_path = _write_summary(summary, output_dir)
    journal.write("autonomous_window_summary", {"summary_md": str(md_path), "summary": asdict(summary)})
    return md_path
