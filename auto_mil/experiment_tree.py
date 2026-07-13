from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .data import prepare_dataset
from .failure_policy import action_to_payload, decide_failure_action, make_retry_recipe
from .mil_baseline import Recipe, RunResult, run_recipe, run_result_from_payload, run_result_to_payload
from .state import ExperimentCheckpoint, ResearchJournal, json_ready, now_iso


@dataclass
class ExperimentNode:
    node_id: str
    parent_id: str | None
    recipe: Recipe
    depth: int
    prior: float
    rationale: str
    status: str = "pending"
    score: float | None = None
    visits: int = 0
    result_run_id: str | None = None
    children: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def selection_score(self, parent_visits: int, c_puct: float) -> float:
        q_value = self.score if self.score is not None and not math.isnan(self.score) else 0.0
        exploration = c_puct * self.prior * math.sqrt(max(parent_visits, 1)) / (1 + self.visits)
        if self.status == "pending":
            return q_value + exploration
        if self.status == "completed":
            return q_value + 0.1 * exploration
        return -1.0 + 0.1 * exploration


class ExperimentTree:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, ExperimentNode] = {}
        self.metadata: dict[str, Any] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.metadata = dict(payload.get("metadata", {}))
        self.nodes = {
            node_id: ExperimentNode(
                node_id=node["node_id"],
                parent_id=node.get("parent_id"),
                recipe=Recipe(**node["recipe"]),
                depth=int(node["depth"]),
                prior=float(node.get("prior", 1.0)),
                rationale=str(node.get("rationale", "")),
                status=str(node.get("status", "pending")),
                score=node.get("score"),
                visits=int(node.get("visits", 0)),
                result_run_id=node.get("result_run_id"),
                children=list(node.get("children", [])),
                created_at=str(node.get("created_at", now_iso())),
                updated_at=str(node.get("updated_at", now_iso())),
            )
            for node_id, node in payload.get("nodes", {}).items()
        }

    def save(self) -> None:
        payload = {
            "version": 1,
            "updated_at": now_iso(),
            "metadata": json_ready(self.metadata),
            "nodes": {node_id: json_ready(asdict(node)) for node_id, node in sorted(self.nodes.items())},
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def add_node(self, node: ExperimentNode) -> bool:
        if node.node_id in self.nodes:
            return False
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children:
                parent.children.append(node.node_id)
                parent.updated_at = now_iso()
        return True

    def update_from_result(self, node_id: str, result: RunResult) -> None:
        node = self.nodes[node_id]
        node.status = result.status
        node.score = result.score if result.score != float("-inf") else None
        node.visits += 1
        node.result_run_id = result.recipe.recipe_id
        node.updated_at = now_iso()

    def pending_nodes(self) -> list[ExperimentNode]:
        return [node for node in self.nodes.values() if node.status == "pending"]

    def completed_nodes(self) -> list[ExperimentNode]:
        return [node for node in self.nodes.values() if node.status == "completed"]

    def select_next(self, c_puct: float = 1.5) -> ExperimentNode | None:
        pending = self.pending_nodes()
        if not pending:
            return None
        total_visits = sum(max(0, node.visits) for node in self.nodes.values()) + 1
        return max(pending, key=lambda node: (node.selection_score(total_visits, c_puct), -node.depth, node.node_id))


def _load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _node_id(*parts: Any) -> str:
    text = "_".join(str(part).lower().replace(".", "p").replace("-", "_") for part in parts)
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)


def _seed_root_nodes(tree: ExperimentTree, cfg: AutoMilConfig, max_screen_models: int | None) -> None:
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    models = list(search.get("screen_models", ["AB_MIL", "TRANS_MIL", "RRT_MIL"]))
    if max_screen_models is not None:
        models = models[:max_screen_models]
    lr = float(search.get("learning_rates", [0.0002])[0])
    dropout = float(search.get("dropouts", [0.1])[0])
    for idx, model in enumerate(models):
        recipe = Recipe(
            recipe_id=_node_id("tree_root", idx, model),
            stage="tree_screen",
            model_name=model,
            epochs=int(training.get("screening_epochs", 1)),
            lr=lr,
            dropout=dropout,
            balanced_sampler=bool(training.get("balanced_sampler", False)),
            notes="QWBE-lite root baseline node.",
        )
        tree.add_node(
            ExperimentNode(
                node_id=recipe.recipe_id,
                parent_id=None,
                recipe=recipe,
                depth=0,
                prior=1.0,
                rationale=f"Screen {model} as a root baseline.",
            )
        )


def _expand_completed_roots(tree: ExperimentTree, cfg: AutoMilConfig, max_children_per_parent: int) -> int:
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    lrs = [float(x) for x in search.get("learning_rates", [0.0002, 0.0001])]
    dropouts = [float(x) for x in search.get("dropouts", [0.1, 0.25])]
    parents = sorted(
        [node for node in tree.completed_nodes() if node.depth == 0],
        key=lambda node: node.score if node.score is not None else float("-inf"),
        reverse=True,
    )
    added = 0
    for parent in parents:
        if len(parent.children) >= max_children_per_parent:
            continue
        combos = [(lr, dropout, True) for lr in lrs for dropout in dropouts]
        for lr, dropout, balanced_sampler in combos:
            if len(parent.children) >= max_children_per_parent:
                break
            recipe = Recipe(
                recipe_id=_node_id("tree_child", parent.recipe.model_name, lr, dropout, int(balanced_sampler)),
                stage="tree_focused",
                model_name=parent.recipe.model_name,
                epochs=int(training.get("focused_epochs", 3)),
                lr=lr,
                dropout=dropout,
                balanced_sampler=balanced_sampler,
                notes=f"QWBE-lite child of {parent.node_id}.",
            )
            child = ExperimentNode(
                node_id=recipe.recipe_id,
                parent_id=parent.node_id,
                recipe=recipe,
                depth=parent.depth + 1,
                prior=0.75,
                rationale=(
                    f"Exploit {parent.recipe.model_name} after parent score "
                    f"{parent.score if parent.score is not None else 'NA'} with lr={lr}, dropout={dropout}."
                ),
            )
            if tree.add_node(child):
                added += 1
    return added


def _add_failure_retry_node(tree: ExperimentTree, failed_node: ExperimentNode, result: RunResult) -> ExperimentNode | None:
    action = decide_failure_action(result.diagnosis)
    if not action.retryable:
        return None
    if failed_node.depth >= int(tree.metadata.get("max_failure_retry_depth", 1)):
        return None
    retry_children = [
        child_id
        for child_id in failed_node.children
        if child_id in tree.nodes and tree.nodes[child_id].recipe.stage.endswith("_retry")
    ]
    if retry_children:
        return None

    retry_recipe = make_retry_recipe(failed_node.recipe, action, retry_index=len(retry_children) + 1)
    if retry_recipe is None:
        return None
    retry_node = ExperimentNode(
        node_id=_node_id(retry_recipe.recipe_id),
        parent_id=failed_node.node_id,
        recipe=retry_recipe,
        depth=failed_node.depth + 1,
        prior=0.35,
        rationale=f"Failure policy retry after {action.category}: {action.summary}",
    )
    if tree.add_node(retry_node):
        return retry_node
    return None


def _write_tree_report(path: Path, cfg: AutoMilConfig, tree: ExperimentTree) -> Path:
    ranked = sorted(
        tree.nodes.values(),
        key=lambda node: node.score if node.score is not None else float("-inf"),
        reverse=True,
    )
    lines = [
        f"# QWBE-Lite Experiment Tree: {cfg.name}",
        "",
        f"Generated: {now_iso()}",
        "",
        "| Rank | Node | Parent | Depth | Model | Status | Score | Prior | Visits | Rationale |",
        "|---:|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, node in enumerate(ranked, start=1):
        score = "" if node.score is None else f"{node.score:.4f}"
        parent = node.parent_id or ""
        rationale = node.rationale.replace("|", "/")
        lines.append(
            f"| {rank} | `{node.node_id}` | `{parent}` | {node.depth} | {node.recipe.model_name} | "
            f"{node.status} | {score} | {node.prior:.2f} | {node.visits} | {rationale} |"
        )
    report_path = path / "experiment_tree.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_qwbe_lite(
    cfg: AutoMilConfig,
    *,
    max_runs: int,
    max_screen_models: int | None = None,
    max_children_per_parent: int = 4,
    max_failure_retry_depth: int = 1,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    output_dir = cfg.output_dir / "experiment_tree"
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")
    tree = ExperimentTree(output_dir / "experiment_tree.json")

    task = cfg.task_spec
    dataset = cfg.dataset_spec
    artifacts = prepare_dataset(
        dataset=dataset,
        task=task,
        output_dir=output_dir / "dataset",
    )
    metadata = _load_metadata(artifacts.metadata_json)
    tree.metadata.update(
        {
            "config": str(cfg.path),
            "dataset_csv": str(artifacts.dataset_csv),
            "metadata_json": str(artifacts.metadata_json),
            "max_runs": max_runs,
            "dry_run": dry_run,
            "max_failure_retry_depth": max_failure_retry_depth,
        }
    )
    checkpoint.update_metadata(
        command="run-tree",
        config=str(cfg.path),
        dataset_csv=str(artifacts.dataset_csv),
        metadata_json=str(artifacts.metadata_json),
        dry_run=dry_run,
    )
    _seed_root_nodes(tree, cfg, max_screen_models)
    tree.save()

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

    executed = 0
    while executed < max_runs:
        if not tree.pending_nodes():
            _expand_completed_roots(tree, cfg, max_children_per_parent)
            tree.save()
        node = tree.select_next(c_puct=float(cfg.raw.get("search", {}).get("tree_c_puct", 1.5)))
        if node is None:
            break

        cached_payload = checkpoint.get_completed_payload(node.recipe.recipe_id) if resume and not dry_run else None
        if cached_payload is not None:
            result = run_result_from_payload(cached_payload)
            journal.write("tree_node_resume", {"node_id": node.node_id, "recipe": asdict(node.recipe)})
        else:
            journal.write("tree_node_start", {"node_id": node.node_id, "recipe": asdict(node.recipe)})
            result = run_recipe(node.recipe, **common)
            checkpoint.record_run(node.recipe.recipe_id, node.recipe.stage, result.status, run_result_to_payload(result))
            journal.write("tree_node_result", {"node_id": node.node_id, "result": run_result_to_payload(result)})

        tree.update_from_result(node.node_id, result)
        if result.status == "failed":
            retry_node = _add_failure_retry_node(tree, node, result)
            action = decide_failure_action(result.diagnosis)
            journal.write(
                "tree_failure_policy",
                {
                    "node_id": node.node_id,
                    "diagnosis": result.diagnosis.category if result.diagnosis else None,
                    "action": action_to_payload(action),
                    "retry_node_id": retry_node.node_id if retry_node else None,
                },
            )
        _expand_completed_roots(tree, cfg, max_children_per_parent)
        tree.save()
        executed += 1

    report = _write_tree_report(output_dir, cfg, tree)
    checkpoint.update_metadata(report=str(report), tree_json=str(tree.path))
    journal.write("tree_report", {"path": str(report), "tree_json": str(tree.path)})
    return report
