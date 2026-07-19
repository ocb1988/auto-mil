from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .baseline_registry import available_models
from .data import _load_h5_classification_slide_table
from .outcome_tasks import VENDORED_OUTCOME_MODELS, load_outcome_task_tables
from .specs import DatasetSpec, TaskSpec
from .state import json_ready


@dataclass(frozen=True)
class BaselineFamilySpec:
    model_name: str
    tier: str
    family: str
    coordinate_policy: str = "not_required"
    dependencies: list[str] = field(default_factory=list)
    memory_risk: str = "medium"
    default_role: str = "candidate"
    notes: str = ""
    pilot_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineAssessment:
    model_name: str
    tier: str
    family: str
    trainable: bool
    compatible: bool
    recommended_for_screen: bool
    coordinate_policy: str
    has_real_coords: bool
    dependency_status: dict[str, bool]
    memory_risk: str
    default_role: str
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    pilot_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselinePlan:
    dataset: str
    feature: dict[str, Any]
    has_real_coords: bool
    recommended_screen: list[str]
    assessments: list[BaselineAssessment]
    warnings: list[str] = field(default_factory=list)


BASELINE_FAMILIES: dict[str, BaselineFamilySpec] = {
    "MEAN_MIL": BaselineFamilySpec("MEAN_MIL", "sanity", "pooling", memory_risk="low", default_role="sanity"),
    "MAX_MIL": BaselineFamilySpec("MAX_MIL", "sanity", "pooling", memory_risk="low", default_role="sanity"),
    "AB_MIL": BaselineFamilySpec("AB_MIL", "classic", "attention", memory_risk="low", default_role="screen"),
    "GATE_AB_MIL": BaselineFamilySpec("GATE_AB_MIL", "classic", "attention", memory_risk="low"),
    "CLAM_SB_MIL": BaselineFamilySpec("CLAM_SB_MIL", "classic", "attention_cluster", memory_risk="medium"),
    "CLAM_MB_MIL": BaselineFamilySpec("CLAM_MB_MIL", "classic", "attention_cluster", memory_risk="medium"),
    "TRANS_MIL": BaselineFamilySpec("TRANS_MIL", "strong_classic", "transformer", memory_risk="high", default_role="screen"),
    "RRT_MIL": BaselineFamilySpec("RRT_MIL", "strong_classic", "long_context", memory_risk="medium", default_role="screen"),
    "STABLE_MIL": BaselineFamilySpec(
        "STABLE_MIL",
        "recent",
        "spatial_stable_attention",
        coordinate_policy="recommended",
        dependencies=["scipy"],
        memory_risk="high",
        default_role="screen",
        notes="Recent spatial MIL; real coords are strongly preferred, pseudo-grid fallback is available in the vendored module.",
        pilot_overrides={"Model.aggregate_num": 64, "Model.k_neighbors": 4},
    ),
    "GDF_MIL": BaselineFamilySpec(
        "GDF_MIL",
        "recent",
        "graph_decomposition",
        memory_risk="medium",
        default_role="screen",
        notes="Recent graph/decomposition baseline; does not require explicit coordinates in the current config.",
        pilot_overrides={"Model.k_components": 5, "Model.k_neighbors": 5},
    ),
    "DAG_MIL": BaselineFamilySpec(
        "DAG_MIL",
        "spatial",
        "dynamic_adjacency_graph",
        coordinate_policy="recommended",
        memory_risk="high",
        notes="Coordinate-aware graph MIL. Feature-only inputs fall back to pseudo-grid in the vendored module.",
        pilot_overrides={"Model.max_instances": 4096},
    ),
    "PSA_MIL": BaselineFamilySpec(
        "PSA_MIL",
        "spatial",
        "spatial_prior_attention",
        coordinate_policy="recommended",
        memory_risk="high",
        notes="Builds spatial distance matrices; cap max_instances for large bags.",
        pilot_overrides={"Model.max_instances": 2048},
    ),
    "SC_MIL": BaselineFamilySpec(
        "SC_MIL",
        "spatial",
        "spatial_clustering",
        coordinate_policy="required",
        dependencies=["sklearn"],
        memory_risk="medium",
        notes="Requires coordinates for clustering; sklearn is sufficient, cuML is optional.",
    ),
    "MAMBA2D_MIL": BaselineFamilySpec(
        "MAMBA2D_MIL",
        "spatial",
        "state_space_2d",
        coordinate_policy="recommended",
        memory_risk="medium",
        notes="2D/grid-oriented sequence model; best used when coordinates or stable grid layout are meaningful.",
    ),
}


DEFAULT_SCREEN_ORDER = ["AB_MIL", "TRANS_MIL", "RRT_MIL", "STABLE_MIL", "GDF_MIL"]
OUTCOME_SCREEN_ORDER = ["AB_MIL", "TRANS_MIL", "RRT_MIL", "STABLE_MIL", "GDF_MIL"]
OUTCOME_SUPPORTED_MODELS = set(VENDORED_OUTCOME_MODELS)
OUTCOME_OPTIONAL_DEPENDENCIES: dict[str, list[str]] = {
    "LONG_MIL": ["xformers"],
    "MAMBA_MIL": ["mamba_ssm"],
    "MAMBA2D_MIL": ["mamba_ssm"],
    "MICRO_MIL": ["dgl"],
    "MSM_MIL": ["mamba_ssm"],
    "WIKG_MIL": ["torch_geometric"],
    "DT_MIL": ["MultiScaleDeformableAttention"],
}


def _outcome_dependency_status(model_name: str, dependencies: list[str]) -> dict[str, bool]:
    status = _dependency_status(dependencies + OUTCOME_OPTIONAL_DEPENDENCIES.get(model_name, []))
    if model_name == "PGCN_MIL":
        status["torch_geometric_or_dgl"] = any(
            importlib.util.find_spec(package) is not None for package in ("torch_geometric", "dgl")
        )
    return status


def _dependency_status(dependencies: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in dependencies}


def _has_real_coords(feature: dict[str, Any]) -> bool:
    if feature.get("coords_shape") is not None:
        return True
    keys = {str(key).lower() for key in feature.get("keys", [])}
    return bool({"coords", "coord", "coordinates", "coords_patching"} & keys)


def _assess_one(
    spec: BaselineFamilySpec,
    *,
    trainable: bool,
    has_real_coords: bool,
    feature: dict[str, Any],
) -> BaselineAssessment:
    dep_status = _dependency_status(spec.dependencies)
    warnings = []
    compatible = trainable and all(dep_status.values())
    if not trainable:
        warnings.append("Missing config/module/process in MIL_BASELINE.")
    missing_deps = [name for name, ok in dep_status.items() if not ok]
    if missing_deps:
        warnings.append("Missing dependencies: " + ", ".join(missing_deps))
    if spec.coordinate_policy == "required" and not has_real_coords:
        compatible = False
        warnings.append("Requires real coordinates, but no coordinate key was detected.")
    elif spec.coordinate_policy == "recommended" and not has_real_coords:
        warnings.append("Real coordinates are recommended; current run may use pseudo-grid behavior.")
    if spec.coordinate_policy in {"required", "recommended"} and has_real_coords and feature.get("coords_shape") is None:
        warnings.append("Coordinate-like keys exist; set dataset.feature.coords_key to use an explicit coordinate dataset.")
    recommended = compatible and spec.model_name in DEFAULT_SCREEN_ORDER
    return BaselineAssessment(
        model_name=spec.model_name,
        tier=spec.tier,
        family=spec.family,
        trainable=trainable,
        compatible=compatible,
        recommended_for_screen=recommended,
        coordinate_policy=spec.coordinate_policy,
        has_real_coords=has_real_coords,
        dependency_status=dep_status,
        memory_risk=spec.memory_risk,
        default_role=spec.default_role,
        warnings=warnings,
        notes=spec.notes,
        pilot_overrides=spec.pilot_overrides,
    )


def build_baseline_plan(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    mil_baseline_dir: str | Path | None = None,
    models: list[str] | None = None,
) -> BaselinePlan:
    if task.kind != "classification":
        tables = load_outcome_task_tables(dataset, task)
        has_coords = _has_real_coords(tables.feature)
        registry = available_models(mil_baseline_dir)
        names = models or OUTCOME_SCREEN_ORDER
        assessments = []
        for name in names:
            spec = BASELINE_FAMILIES.get(name, BaselineFamilySpec(name, "other", "unclassified"))
            dep_status = _outcome_dependency_status(name, spec.dependencies)
            is_native = name in {"MEAN_MIL", "MAX_MIL", "GATE_AB_MIL"}
            vendored = registry.get(name)
            has_vendored_module = bool(vendored and vendored.has_module)
            compatible = name in OUTCOME_SUPPORTED_MODELS and (is_native or has_vendored_module) and all(dep_status.values())
            warnings = [] if compatible else ["Outcome adapter does not support this model or a required runtime dependency is unavailable."]
            if name in OUTCOME_SCREEN_ORDER and not has_vendored_module:
                warnings.append("The selected MIL_BASELINE directory does not contain this model module.")
            if spec.coordinate_policy == "recommended" and not has_coords:
                warnings.append("Real coordinates are recommended; this model will use its vendored pseudo-grid fallback.")
            assessments.append(
                BaselineAssessment(
                    model_name=name,
                    tier=spec.tier,
                    family=spec.family,
                    trainable=compatible,
                    compatible=compatible,
                    recommended_for_screen=name in OUTCOME_SCREEN_ORDER and compatible,
                    coordinate_policy="not_required",
                    has_real_coords=has_coords,
                    dependency_status=dep_status,
                    memory_risk=spec.memory_risk,
                    default_role=spec.default_role,
                    warnings=warnings,
                    notes="Uses the vendored MIL encoder/aggregator with Auto-MIL's task-specific head and outcome loss, not the classification-only MIL_BASELINE trainer.",
                    pilot_overrides={},
                )
            )
        return BaselinePlan(
            dataset=dataset.name,
            feature=tables.feature,
            has_real_coords=has_coords,
            recommended_screen=[name for name in OUTCOME_SCREEN_ORDER if any(item.model_name == name and item.compatible for item in assessments)],
            assessments=assessments,
            warnings=["Outcome runs reuse supported vendored MIL aggregators but replace their classification head and loss with task-specific components."],
        )
    _slide_df, _case_df, _label_to_id, _missing, _total, feature = _load_h5_classification_slide_table(dataset, task)
    has_coords = _has_real_coords(feature)
    registry = available_models(mil_baseline_dir)
    names = models or sorted(BASELINE_FAMILIES)
    assessments = []
    warnings = []
    for name in names:
        spec = BASELINE_FAMILIES.get(
            name,
            BaselineFamilySpec(name, "other", "unclassified", notes="No curated family metadata yet."),
        )
        info = registry.get(name)
        assessments.append(
            _assess_one(spec, trainable=bool(info and info.is_trainable), has_real_coords=has_coords, feature=feature)
        )
    recommended = [name for name in DEFAULT_SCREEN_ORDER if any(a.model_name == name and a.compatible for a in assessments)]
    if len(recommended) < 4:
        warnings.append("Fewer than four preferred screen baselines are compatible; inspect warnings before running.")
    return BaselinePlan(
        dataset=dataset.name,
        feature=feature,
        has_real_coords=has_coords,
        recommended_screen=recommended,
        assessments=assessments,
        warnings=warnings,
    )


def baseline_plan_to_payload(plan: BaselinePlan) -> dict[str, Any]:
    return json_ready(asdict(plan))


def write_baseline_plan(plan: BaselinePlan, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_plan.json"
    md_path = output_dir / "baseline_plan.md"
    payload = baseline_plan_to_payload(plan)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(plan), encoding="utf-8")
    return json_path, md_path


def _render_markdown(plan: BaselinePlan) -> str:
    lines = [
        f"# Baseline Plan: {plan.dataset}",
        "",
        f"- Has coordinate-like data: `{plan.has_real_coords}`",
        f"- Recommended screen: `{plan.recommended_screen}`",
        f"- Feature: `{plan.feature}`",
        "",
        "| Model | Tier | Family | Trainable | Compatible | Screen | Coords | Memory | Warnings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in plan.assessments:
        warnings = "<br>".join(item.warnings)
        lines.append(
            f"| {item.model_name} | {item.tier} | {item.family} | {str(item.trainable).lower()} | "
            f"{str(item.compatible).lower()} | {str(item.recommended_for_screen).lower()} | "
            f"{item.coordinate_policy} | {item.memory_risk} | {warnings} |"
        )
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in plan.warnings])
    lines.extend(
        [
            "",
            "## Confirmation Gate",
            "",
            "Confirm the baseline suite and budget before launching manuscript-grade runs.",
        ]
    )
    return "\n".join(lines)
