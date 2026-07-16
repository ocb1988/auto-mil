from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoMilConfig
from .experiment_tree import ExperimentNode, ExperimentTree
from .innovation_policy import classify_recipe_stage
from .mil_baseline import Recipe
from .state import json_ready, now_iso


@dataclass(frozen=True)
class ProposalRecord:
    node_id: str
    model_name: str
    proposal_type: str
    prior: float
    added: bool
    rationale: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    innovation_track: str = "baseline"
    core_modules: list[str] = field(default_factory=list)
    support_tags: list[str] = field(default_factory=list)
    ablation_plan: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProposalSummary:
    generated_at: str
    tree_json: str
    checkpoint: str
    baseline_plan: str | None
    max_proposals: int
    applied: bool
    proposals: list[ProposalRecord]
    warnings: list[str] = field(default_factory=list)


def _safe_id(*parts: Any) -> str:
    text = "_".join(str(part).lower().replace(".", "p").replace("-", "_") for part in parts)
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_models(tree: ExperimentTree) -> set[str]:
    return {node.recipe.model_name for node in tree.nodes.values()}


def _best_completed_recipe(checkpoint_path: Path) -> tuple[Recipe | None, float | None]:
    checkpoint = _load_json(checkpoint_path)
    best_recipe = None
    best_score = None
    for record in checkpoint.get("runs", {}).values():
        if record.get("status") != "completed":
            continue
        payload = record.get("payload", {})
        recipe_payload = payload.get("recipe")
        if not isinstance(recipe_payload, dict):
            continue
        metrics = payload.get("metrics", {})
        value = metrics.get("test_macro_auc", metrics.get("val_macro_auc"))
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        recipe_payload = dict(recipe_payload)
        recipe_payload.setdefault("config_overrides", {})
        recipe = Recipe(**recipe_payload)
        if best_score is None or score > best_score:
            best_score = score
            best_recipe = recipe
    return best_recipe, best_score


def _family_candidates(
    cfg: AutoMilConfig,
    tree: ExperimentTree,
    baseline_plan: dict[str, Any],
    max_candidates: int,
) -> list[ExperimentNode]:
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    lr = float(search.get("learning_rates", [0.0002])[0])
    dropout = float(search.get("dropouts", [0.1])[0])
    existing = _existing_models(tree)
    assessments = list(baseline_plan.get("assessments", []))
    recommended = list(baseline_plan.get("recommended_screen", []))
    ordered_names = recommended + [
        str(item.get("model_name"))
        for item in assessments
        if item.get("compatible") and item.get("tier") in {"spatial", "recent"}
    ]
    nodes: list[ExperimentNode] = []
    seen: set[str] = set()
    for model in ordered_names:
        if model in seen or model in existing:
            continue
        seen.add(model)
        assessment = next((item for item in assessments if item.get("model_name") == model), {})
        overrides = dict(assessment.get("pilot_overrides") or {})
        node_id = _safe_id("proposal", "family", model)
        recipe = Recipe(
            recipe_id=node_id,
            stage="proposal_family",
            model_name=model,
            epochs=int(training.get("screening_epochs", 1)),
            lr=lr,
            dropout=dropout,
            balanced_sampler=bool(training.get("balanced_sampler", False)),
            notes=f"Proposal generator family screen for {model}.",
            config_overrides=overrides,
        )
        prior = 0.85 if model in recommended else 0.55
        rationale = (
            f"Add compatible {assessment.get('tier', 'candidate')} baseline {model}; "
            f"family={assessment.get('family', 'unknown')}, memory={assessment.get('memory_risk', 'unknown')}."
        )
        nodes.append(
            ExperimentNode(
                node_id=node_id,
                parent_id=None,
                recipe=recipe,
                depth=0,
                prior=prior,
                rationale=rationale,
                innovation_track="baseline",
            )
        )
        if len(nodes) >= max_candidates:
            break
    return nodes


def _exploit_candidates(cfg: AutoMilConfig, tree: ExperimentTree, checkpoint_path: Path, max_candidates: int) -> list[ExperimentNode]:
    best_recipe, best_score = _best_completed_recipe(checkpoint_path)
    if best_recipe is None:
        return []
    training = cfg.raw.get("training", {})
    specs = [
        ("lr_half", max(best_recipe.lr * 0.5, 1e-7), best_recipe.dropout, True, 0.78),
        ("lr_quarter", max(best_recipe.lr * 0.25, 1e-7), best_recipe.dropout, best_recipe.balanced_sampler, 0.65),
        (
            "dropout_high",
            best_recipe.lr,
            min(0.5, (best_recipe.dropout if best_recipe.dropout is not None else 0.1) + 0.15),
            True,
            0.62,
        ),
    ]
    parent = next((node for node in tree.nodes.values() if node.recipe.recipe_id == best_recipe.recipe_id), None)
    nodes: list[ExperimentNode] = []
    for suffix, lr, dropout, balanced, prior in specs:
        node_id = _safe_id("proposal", "exploit", best_recipe.model_name, suffix)
        recipe = Recipe(
            recipe_id=node_id,
            stage="proposal_exploit",
            model_name=best_recipe.model_name,
            epochs=int(training.get("focused_epochs", max(best_recipe.epochs, 3))),
            lr=lr,
            dropout=dropout,
            balanced_sampler=balanced,
            notes=f"Proposal generator local refinement from {best_recipe.recipe_id}.",
            config_overrides=dict(best_recipe.config_overrides),
        )
        nodes.append(
            ExperimentNode(
                node_id=node_id,
                parent_id=parent.node_id if parent else None,
                recipe=recipe,
                depth=(parent.depth + 1) if parent else 1,
                prior=prior,
                rationale=(
                    f"Exploit best completed recipe {best_recipe.recipe_id} "
                    f"(score={best_score}) with {suffix}."
                ),
                innovation_track="support",
                support_tags=["learning-rate tuning", "dropout tuning", "sampler tuning"],
            )
        )
        if len(nodes) >= max_candidates:
            break
    return nodes


def _method_candidates(cfg: AutoMilConfig, tree: ExperimentTree, max_candidates: int) -> list[ExperimentNode]:
    innovation = cfg.raw.get("innovation", {})
    proposals = innovation.get("method_proposals", [])
    if not proposals:
        proposals = _load_literature_method_proposals(cfg)
    if not isinstance(proposals, list):
        return []
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    lr = float(search.get("learning_rates", [0.0002])[0])
    dropout = float(innovation.get("dropout", search.get("dropouts", [0.25])[0]))
    existing = set(tree.nodes)
    nodes: list[ExperimentNode] = []
    for idx, item in enumerate(proposals):
        if not isinstance(item, dict):
            continue
        model = str(item.get("model_name", "AB_MIL"))
        name = str(item.get("name", f"method_{idx + 1}"))
        node_id = _safe_id("proposal", "method", name)
        if node_id in existing:
            continue
        core_modules = [str(x) for x in item.get("core_modules", [])]
        if not core_modules:
            continue
        support_tags = [str(x) for x in item.get("support_tags", [])]
        ablation_plan = [str(x) for x in item.get("ablation_plan", core_modules)]
        overrides = dict(item.get("config_overrides") or {})
        recipe = Recipe(
            recipe_id=node_id,
            stage="proposal_method",
            model_name=model,
            epochs=int(item.get("epochs", training.get("focused_epochs", 3))),
            lr=float(item.get("lr", lr)),
            dropout=float(item.get("dropout", dropout)),
            balanced_sampler=bool(item.get("balanced_sampler", innovation.get("balanced_sampler", True))),
            notes=str(item.get("rationale", f"Method proposal with core modules: {', '.join(core_modules)}.")),
            config_overrides=overrides,
        )
        nodes.append(
            ExperimentNode(
                node_id=node_id,
                parent_id=None,
                recipe=recipe,
                depth=0,
                prior=float(item.get("prior", 0.9)),
                rationale=str(item.get("rationale", "Configured method proposal.")),
                innovation_track="method",
                core_modules=core_modules,
                support_tags=support_tags,
                ablation_plan=ablation_plan,
            )
        )
        if len(nodes) >= max_candidates:
            break
    return nodes


def _load_literature_method_proposals(cfg: AutoMilConfig) -> list[dict[str, Any]]:
    settings = cfg.raw.get("innovation", {}).get("literature_search", {})
    proposal_path = settings.get("proposals_json") or cfg.output_dir / "literature_search" / "literature_proposals.json"
    path = Path(proposal_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        return []
    out: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": item.get("name"),
                "model_name": item.get("model_name", "AB_MIL"),
                "core_modules": item.get("core_modules", []),
                "support_tags": item.get("support_tags", []),
                "ablation_plan": item.get("ablation_plan", []),
                "rationale": item.get("motivation") or item.get("mechanism") or "Literature-derived method proposal.",
                "config_overrides": item.get("config_overrides", {}),
            }
        )
    return out


def propose_nodes(
    cfg: AutoMilConfig,
    *,
    tree_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    baseline_plan_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_proposals: int = 6,
    apply: bool = True,
) -> Path:
    tree_path = Path(tree_path) if tree_path else cfg.output_dir / "experiment_tree" / "experiment_tree.json"
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else cfg.output_dir / "experiment_tree" / "checkpoint.json"
    baseline_plan_path = Path(baseline_plan_path) if baseline_plan_path else cfg.output_dir / "baseline_plan" / "baseline_plan.json"
    output_dir = Path(output_dir) if output_dir else cfg.output_dir / "proposal_generator"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ExperimentTree(tree_path)
    baseline_plan = _load_json(baseline_plan_path)
    warnings = []
    if not baseline_plan:
        warnings.append(f"Baseline plan not found or empty: {baseline_plan_path}")

    candidates: list[ExperimentNode] = []
    candidates.extend(_method_candidates(cfg, tree, max_proposals))
    if baseline_plan:
        candidates.extend(_family_candidates(cfg, tree, baseline_plan, max_proposals - len(candidates)))
    remaining = max(0, max_proposals - len(candidates))
    candidates.extend(_exploit_candidates(cfg, tree, checkpoint_path, remaining))

    records: list[ProposalRecord] = []
    for node in candidates[:max_proposals]:
        added = tree.add_node(node) if apply else False
        records.append(
            _record_from_node(node, added)
        )
    if not records:
        warnings.append("No new proposal candidates were generated from the current tree and evidence.")
    if apply:
        tree.metadata["last_proposal_generated_at"] = now_iso()
        tree.metadata["last_proposal_count"] = len([record for record in records if record.added])
        tree.save()

    summary = ProposalSummary(
        generated_at=now_iso(),
        tree_json=str(tree_path),
        checkpoint=str(checkpoint_path),
        baseline_plan=str(baseline_plan_path) if baseline_plan_path else None,
        max_proposals=max_proposals,
        applied=apply,
        proposals=records,
        warnings=warnings,
    )
    json_path = output_dir / "proposal_report.json"
    md_path = output_dir / "proposal_report.md"
    payload = json_ready(asdict(summary))
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    return md_path


def _record_from_node(node: ExperimentNode, added: bool) -> ProposalRecord:
    policy = classify_recipe_stage(node.recipe.stage, node.recipe.notes, node.recipe.config_overrides)
    track = node.innovation_track or policy.track
    core_modules = list(node.core_modules or policy.core_modules)
    support_tags = list(node.support_tags or policy.support_tags)
    return ProposalRecord(
        node_id=node.node_id,
        model_name=node.recipe.model_name,
        proposal_type=node.recipe.stage,
        prior=node.prior,
        added=added,
        rationale=node.rationale,
        config_overrides=dict(node.recipe.config_overrides),
        innovation_track=track,
        core_modules=core_modules,
        support_tags=support_tags,
        ablation_plan=list(node.ablation_plan),
    )


def _render_markdown(summary: ProposalSummary) -> str:
    lines = [
        "# Proposal Report",
        "",
        f"- Generated: `{summary.generated_at}`",
        f"- Applied: `{summary.applied}`",
        f"- Tree: `{summary.tree_json}`",
        f"- Checkpoint: `{summary.checkpoint}`",
        f"- Baseline plan: `{summary.baseline_plan}`",
        "",
        "| Added | Node | Track | Type | Model | Core modules | Support tags | Ablation plan | Prior | Overrides | Rationale |",
        "|---|---|---|---|---|---|---|---|---:|---|---|",
    ]
    for item in summary.proposals:
        rationale = item.rationale.replace("|", "/")
        core = ", ".join(item.core_modules)
        support = ", ".join(item.support_tags)
        ablation = ", ".join(item.ablation_plan)
        lines.append(
            f"| {str(item.added).lower()} | `{item.node_id}` | {item.innovation_track} | {item.proposal_type} | "
            f"{item.model_name} | {core} | {support} | {ablation} | {item.prior:.2f} | `{item.config_overrides}` | {rationale} |"
        )
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary.warnings])
    return "\n".join(lines)
