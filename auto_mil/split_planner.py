from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data import _load_h5_classification_slide_table
from .outcome_tasks import load_outcome_task_tables
from .specs import DatasetSpec, TaskSpec, specs_to_payload
from .state import json_ready, now_iso


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    task_kind: str
    outcome_column: str
    num_cases: int
    num_slides: int
    num_classes: int | None
    class_counts: dict[str, int]
    center_counts: dict[str, int] = field(default_factory=dict)
    cohort_counts: dict[str, int] = field(default_factory=dict)
    external_test_counts: dict[str, int] = field(default_factory=dict)
    rarest_class_count: int | None = None
    min_cases_per_center: int | None = None
    feature: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitPlanOption:
    plan_id: str
    strategy: str
    recommended: bool
    confirmation_required: bool
    split_unit: str
    rationale: str
    n_splits: int | None = None
    train_centers: list[str] = field(default_factory=list)
    val_centers: list[str] = field(default_factory=list)
    test_centers: list[str] = field(default_factory=list)
    external_test_values: list[str] = field(default_factory=list)
    expected_case_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SplitPlanBundle:
    generated_at: str
    profile: DatasetProfile
    plans: list[SplitPlanOption]
    task: dict[str, Any]
    dataset: dict[str, Any]
    global_warnings: list[str] = field(default_factory=list)


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.astype(str).value_counts().sort_index().items()}


def _label_split_counts(case_df: pd.DataFrame, split_by_case: dict[str, str]) -> dict[str, dict[str, int]]:
    temp = case_df.copy()
    temp["split"] = temp["case_id"].astype(str).map(split_by_case)
    temp = temp.dropna(subset=["split"])
    if temp.empty:
        return {}
    table = temp.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    return {
        str(split): {str(label): int(value) for label, value in row.items()}
        for split, row in table.iterrows()
    }


def _profile_from_case_table(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    slide_df: pd.DataFrame,
    case_df: pd.DataFrame,
    label_to_id: dict[str, int],
    feature: dict[str, Any],
) -> DatasetProfile:
    center_counts = _counts(case_df[dataset.center_column]) if dataset.center_column and dataset.center_column in case_df else {}
    cohort_counts = _counts(case_df[dataset.cohort_column]) if dataset.cohort_column and dataset.cohort_column in case_df else {}
    external_counts = (
        _counts(case_df[dataset.external_test_column])
        if dataset.external_test_column and dataset.external_test_column in case_df
        else {}
    )
    class_counts = _counts(case_df["label_name"]) if "label_name" in case_df else {}
    return DatasetProfile(
        dataset=dataset.name,
        task_kind=task.kind,
        outcome_column=task.outcome_column,
        num_cases=int(case_df.shape[0]),
        num_slides=int(slide_df.shape[0]),
        num_classes=len(label_to_id) if label_to_id else None,
        class_counts=class_counts,
        center_counts=center_counts,
        cohort_counts=cohort_counts,
        external_test_counts=external_counts,
        rarest_class_count=min(class_counts.values()) if class_counts else None,
        min_cases_per_center=min(center_counts.values()) if center_counts else None,
        feature=feature,
    )


def _stratified_cv_plan(profile: DatasetProfile, task: TaskSpec) -> SplitPlanOption:
    rarest = profile.rarest_class_count or 0
    n_splits = max(2, min(5, rarest)) if rarest >= 2 else None
    warnings = []
    if rarest < 5:
        warnings.append(f"Rarest class has {rarest} cases; use fewer than 5 folds or report instability.")
    if n_splits is None:
        warnings.append("Rarest class has fewer than 2 cases; stratified CV is not feasible.")
    return SplitPlanOption(
        plan_id="patient_stratified_cv",
        strategy="n_fold_cross_validation",
        recommended=not profile.center_counts and not profile.external_test_counts,
        confirmation_required=True,
        split_unit="case",
        n_splits=n_splits,
        rationale=(
            "Use patient-level outcome-stratified cross validation when only one dataset/center is available. "
            "For regression, strata are outcome quantiles; for survival, event status is preserved where feasible."
            if task.kind != "classification"
            else "Use patient-level stratified cross validation when only one dataset/center is available. This is the default manuscript-grade choice for a single cohort."
        ),
        warnings=warnings,
    )


def _external_test_plan(case_df: pd.DataFrame, dataset: DatasetSpec) -> SplitPlanOption | None:
    column = dataset.external_test_column
    if not column or column not in case_df.columns:
        return None
    values = sorted(str(x) for x in case_df[column].dropna().astype(str).unique())
    if len(values) < 2:
        return None
    test_values = [value for value in values if value.lower() in {"test", "external", "external_test", "heldout", "holdout"}]
    if not test_values:
        test_values = [values[-1]]
    split_by_case = {
        str(row["case_id"]): "test" if str(row[column]) in test_values else "train_val"
        for _, row in case_df.iterrows()
    }
    return SplitPlanOption(
        plan_id="external_test_holdout",
        strategy="external_test",
        recommended=True,
        confirmation_required=True,
        split_unit="case",
        rationale=(
            f"Use `{column}` to keep predefined external/held-out cases untouched for final testing. "
            "Train/validation should be split only inside the remaining cases."
        ),
        external_test_values=test_values,
        expected_case_counts=_label_split_counts(case_df, split_by_case),
    )


def _center_holdout_plan(case_df: pd.DataFrame, dataset: DatasetSpec) -> SplitPlanOption | None:
    column = dataset.center_column
    if not column or column not in case_df.columns:
        return None
    center_counts = _counts(case_df[column])
    if len(center_counts) < 2:
        return None
    test_center = sorted(center_counts.items(), key=lambda item: (item[1], item[0]))[-1][0]
    train_centers = [center for center in sorted(center_counts) if center != test_center]
    split_by_case = {
        str(row["case_id"]): "test" if str(row[column]) == test_center else "train_val"
        for _, row in case_df.iterrows()
    }
    warnings = []
    if min(center_counts.values()) < 10:
        warnings.append("At least one center has fewer than 10 cases; center holdout may be unstable.")
    return SplitPlanOption(
        plan_id="center_holdout",
        strategy="center_external_test",
        recommended=True,
        confirmation_required=True,
        split_unit="case",
        train_centers=train_centers,
        test_centers=[test_center],
        rationale=(
            f"Use `{column}` for center-aware evaluation: train/validate on one or more centers "
            "and reserve a held-out center as external test."
        ),
        expected_case_counts=_label_split_counts(case_df, split_by_case),
        warnings=warnings,
    )


def _single_holdout_plan(case_df: pd.DataFrame, task: TaskSpec) -> SplitPlanOption:
    warnings = []
    if abs(task.train_size + task.val_size + task.test_size - 1.0) > 1e-6:
        warnings.append("Configured train/val/test ratios do not sum to 1.0.")
    if case_df["label_name"].astype(str).value_counts().min() < 2:
        warnings.append(
            "At least one outcome stratum has fewer than 2 cases; stratified holdout may fail."
            if task.kind != "classification"
            else "At least one class has fewer than 2 cases; stratified holdout may fail."
        )
    return SplitPlanOption(
        plan_id="patient_stratified_holdout",
        strategy="train_val_test_holdout",
        recommended=False,
        confirmation_required=True,
        split_unit="case",
        rationale=(
            "Use a patient-level outcome-stratified train/validation/test split for quick pilots or demos. "
            "For manuscript-grade single-cohort evidence, prefer n-fold CV."
            if task.kind != "classification"
            else "Use a patient-level stratified train/validation/test split for quick pilots or demos. For manuscript-grade single-cohort evidence, prefer n-fold CV."
        ),
        expected_case_counts={
            "ratios": {
                "train": int(round(task.train_size * 100)),
                "val": int(round(task.val_size * 100)),
                "test": int(round(task.test_size * 100)),
            }
        },
        warnings=warnings,
    )


def plan_splits(dataset: DatasetSpec, task: TaskSpec) -> SplitPlanBundle:
    if task.kind == "classification":
        slide_df, case_df, label_to_id, _missing_cases, _num_h5_total, feature = _load_h5_classification_slide_table(
            dataset,
            task,
        )
    else:
        outcome = load_outcome_task_tables(dataset, task)
        slide_df, case_df, label_to_id, feature = outcome.slide_df, outcome.case_df, {}, outcome.feature
    profile = _profile_from_case_table(
        dataset=dataset,
        task=task,
        slide_df=slide_df,
        case_df=case_df,
        label_to_id=label_to_id,
        feature=feature,
    )

    plans: list[SplitPlanOption] = []
    external = _external_test_plan(case_df, dataset)
    center = _center_holdout_plan(case_df, dataset)
    if external is not None:
        plans.append(external)
    if center is not None:
        plans.append(center)
    plans.append(_stratified_cv_plan(profile, task))
    plans.append(_single_holdout_plan(case_df, task))

    if external is None and center is None:
        plans = [
            plan if plan.strategy != "n_fold_cross_validation" else dataclass_replace_recommended(plan, True)
            for plan in plans
        ]

    global_warnings = []
    if profile.rarest_class_count is not None and profile.rarest_class_count < task.min_class_count:
        global_warnings.append(
            "Some outcome strata are below min_class_count after matching slides to labels."
            if task.kind != "classification"
            else "Some classes are below min_class_count after matching slides to labels."
        )
    if profile.num_cases < 50:
        global_warnings.append("Dataset has fewer than 50 cases; use pilot claims cautiously.")

    specs = specs_to_payload(task, dataset)
    return SplitPlanBundle(
        generated_at=now_iso(),
        profile=profile,
        plans=plans,
        task=specs["task"],
        dataset=specs["dataset"],
        global_warnings=global_warnings,
    )


def dataclass_replace_recommended(plan: SplitPlanOption, recommended: bool) -> SplitPlanOption:
    payload = asdict(plan)
    payload["recommended"] = recommended
    return SplitPlanOption(**payload)


def split_plan_to_payload(bundle: SplitPlanBundle) -> dict[str, Any]:
    return json_ready(asdict(bundle))


def write_split_plan(bundle: SplitPlanBundle, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_plan.json"
    md_path = output_dir / "split_plan.md"
    payload = split_plan_to_payload(bundle)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(bundle), encoding="utf-8")
    return json_path, md_path


def _render_markdown(bundle: SplitPlanBundle) -> str:
    profile = bundle.profile
    lines = [
        f"# Split Plan: {profile.dataset}",
        "",
        f"Generated: {bundle.generated_at}",
        "",
        "## Dataset Profile",
        "",
        f"- Task: `{profile.task_kind}`",
        f"- Outcome: `{profile.outcome_column}`",
        f"- Cases: `{profile.num_cases}`",
        f"- Slides: `{profile.num_slides}`",
        f"- Classes: `{profile.class_counts}`",
        f"- Centers: `{profile.center_counts}`",
        f"- Cohorts: `{profile.cohort_counts}`",
        f"- External-test groups: `{profile.external_test_counts}`",
        f"- Feature: `{profile.feature}`",
        "",
        "## Candidate Plans",
        "",
        "| Recommended | Plan | Strategy | Unit | Folds | Rationale | Warnings |",
        "|---|---|---|---|---:|---|---|",
    ]
    for plan in bundle.plans:
        warnings = "<br>".join(plan.warnings)
        rationale = plan.rationale.replace("|", "/")
        lines.append(
            f"| {'yes' if plan.recommended else 'no'} | `{plan.plan_id}` | {plan.strategy} | "
            f"{plan.split_unit} | {plan.n_splits or ''} | {rationale} | {warnings} |"
        )
    if bundle.global_warnings:
        lines.extend(["", "## Global Warnings", ""])
        lines.extend([f"- {warning}" for warning in bundle.global_warnings])
    lines.extend(
        [
            "",
            "## Confirmation Gate",
            "",
            "Confirm one split plan before running manuscript-grade baselines. Do not change the split after seeing test results.",
        ]
    )
    return "\n".join(lines)
