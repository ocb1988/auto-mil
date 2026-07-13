from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import (
    DatasetArtifacts,
    KFoldArtifacts,
    _load_h5_classification_slide_table,
    prepare_dataset_from_case_splits,
    prepare_dataset_kfold_from_case_splits,
)
from .specs import DatasetSpec, TaskSpec


@dataclass(frozen=True)
class SelectedSplitPlan:
    split_plan_path: Path
    plan: dict[str, Any]
    payload: dict[str, Any]

    @property
    def plan_id(self) -> str:
        return str(self.plan["plan_id"])

    @property
    def strategy(self) -> str:
        return str(self.plan["strategy"])


def load_split_plan(path: str | Path) -> dict[str, Any]:
    split_plan_path = Path(path)
    with split_plan_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_split_plan(path: str | Path, plan_id: str | None = None) -> SelectedSplitPlan:
    split_plan_path = Path(path)
    payload = load_split_plan(split_plan_path)
    plans = list(payload.get("plans", []))
    if not plans:
        raise ValueError(f"No plans found in {split_plan_path}")
    if plan_id:
        matches = [plan for plan in plans if str(plan.get("plan_id")) == plan_id]
        if not matches:
            raise ValueError(f"Plan {plan_id!r} was not found in {split_plan_path}")
        return SelectedSplitPlan(split_plan_path=split_plan_path, plan=matches[0], payload=payload)
    recommended = [plan for plan in plans if bool(plan.get("recommended"))]
    if len(recommended) == 1:
        return SelectedSplitPlan(split_plan_path=split_plan_path, plan=recommended[0], payload=payload)
    if len(recommended) > 1:
        ids = ", ".join(str(plan.get("plan_id")) for plan in recommended)
        raise ValueError(f"Multiple recommended split plans exist ({ids}); pass --plan-id explicitly.")
    raise ValueError("No recommended split plan exists; pass --plan-id explicitly.")


def _split_metadata(selected: SelectedSplitPlan) -> dict[str, Any]:
    return {
        "confirmed_split": {
            "split_plan_path": str(selected.split_plan_path),
            "plan_id": selected.plan_id,
            "strategy": selected.strategy,
            "plan": selected.plan,
        }
    }


def _stratified_kfold_case_splits(case_df: pd.DataFrame, task: TaskSpec, n_splits: int) -> list[dict[str, str]]:
    min_class = int(case_df["label_name"].value_counts().min())
    if n_splits > min_class:
        raise ValueError(f"n_splits={n_splits} is too high for the rarest class with {min_class} cases")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=task.split_seed)
    case_df = case_df.reset_index(drop=True)
    fold_splits: list[dict[str, str]] = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(splitter.split(case_df["case_id"], case_df["label_name"])):
        trainval_cases = case_df.iloc[trainval_idx].copy()
        test_cases = case_df.iloc[test_idx].copy()
        stratify = trainval_cases["label_name"]
        if stratify.value_counts().min() < 2:
            stratify = None
        train_cases, val_cases = train_test_split(
            trainval_cases,
            test_size=task.cv_val_fraction_of_train,
            random_state=task.split_seed + fold_idx,
            stratify=stratify,
        )
        fold_splits.append(
            {
                **{case_id: "train" for case_id in train_cases["case_id"].astype(str)},
                **{case_id: "val" for case_id in val_cases["case_id"].astype(str)},
                **{case_id: "test" for case_id in test_cases["case_id"].astype(str)},
            }
        )
    return fold_splits


def materialize_kfold_from_split_plan(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    split_plan_path: str | Path,
    plan_id: str | None = None,
) -> KFoldArtifacts:
    selected = select_split_plan(split_plan_path, plan_id)
    if selected.strategy != "n_fold_cross_validation":
        raise ValueError(f"Plan {selected.plan_id} has strategy {selected.strategy}; use a CV split plan for run-cv.")
    n_splits = selected.plan.get("n_splits")
    if not n_splits:
        raise ValueError(f"Plan {selected.plan_id} does not define n_splits.")
    _slide_df, case_df, _label_to_id, _missing, _total, _feature = _load_h5_classification_slide_table(dataset, task)
    fold_splits = _stratified_kfold_case_splits(case_df, task, int(n_splits))
    return prepare_dataset_kfold_from_case_splits(
        dataset=dataset,
        task=task,
        output_dir=output_dir,
        fold_split_by_case=fold_splits,
        extra_metadata=_split_metadata(selected),
    )


def _stratified_train_val_split(case_df: pd.DataFrame, task: TaskSpec, seed_offset: int = 0) -> tuple[set[str], set[str]]:
    labels = case_df["label_name"].astype(str)
    stratify = labels if labels.value_counts().min() >= 2 else None
    val_fraction = task.val_size / max(task.train_size + task.val_size, 1e-12)
    train_df, val_df = train_test_split(
        case_df,
        test_size=val_fraction,
        random_state=task.split_seed + seed_offset,
        stratify=stratify,
    )
    return set(train_df["case_id"].astype(str)), set(val_df["case_id"].astype(str))


def _holdout_case_split(case_df: pd.DataFrame, selected: SelectedSplitPlan, dataset: DatasetSpec, task: TaskSpec) -> dict[str, str]:
    plan = selected.plan
    if selected.strategy == "external_test":
        column = dataset.external_test_column
        values = {str(value) for value in plan.get("external_test_values", [])}
        if not column or column not in case_df.columns or not values:
            raise ValueError("External-test plan requires dataset.external_test_column and external_test_values.")
        test_cases = set(case_df.loc[case_df[column].astype(str).isin(values), "case_id"].astype(str))
        trainval_df = case_df.loc[~case_df["case_id"].astype(str).isin(test_cases)].copy()
    elif selected.strategy == "center_external_test":
        column = dataset.center_column
        centers = {str(value) for value in plan.get("test_centers", [])}
        if not column or column not in case_df.columns or not centers:
            raise ValueError("Center-holdout plan requires dataset.center_column and test_centers.")
        test_cases = set(case_df.loc[case_df[column].astype(str).isin(centers), "case_id"].astype(str))
        trainval_df = case_df.loc[~case_df["case_id"].astype(str).isin(test_cases)].copy()
    elif selected.strategy == "train_val_test_holdout":
        labels = case_df["label_name"].astype(str)
        stratify = labels if labels.value_counts().min() >= 2 else None
        trainval_df, test_df = train_test_split(
            case_df,
            train_size=task.train_size + task.val_size,
            random_state=task.split_seed,
            stratify=stratify,
        )
        test_cases = set(test_df["case_id"].astype(str))
    else:
        raise ValueError(f"Plan {selected.plan_id} has strategy {selected.strategy}; use run-cv for CV plans.")

    if trainval_df.empty or not test_cases:
        raise ValueError(f"Plan {selected.plan_id} produced an empty train/val or test split.")
    train_cases, val_cases = _stratified_train_val_split(trainval_df, task)
    return {
        **{case_id: "train" for case_id in train_cases},
        **{case_id: "val" for case_id in val_cases},
        **{case_id: "test" for case_id in test_cases},
    }


def materialize_holdout_from_split_plan(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    split_plan_path: str | Path,
    plan_id: str | None = None,
) -> DatasetArtifacts:
    selected = select_split_plan(split_plan_path, plan_id)
    loaded = _load_h5_classification_slide_table(dataset, task)
    _slide_df, case_df, _label_to_id, _missing, _total, _feature = loaded
    split_by_case = _holdout_case_split(case_df, selected, dataset, task)
    return prepare_dataset_from_case_splits(
        dataset=dataset,
        task=task,
        output_dir=output_dir,
        split_by_case=split_by_case,
        extra_metadata=_split_metadata(selected),
        loaded=loaded,
    )
