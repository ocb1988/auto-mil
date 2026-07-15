from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .state import json_ready


TaskKind = Literal["classification", "prognosis", "survival", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    kind: TaskKind
    label_column: str | None = None
    label_threshold: float | None = None
    label_threshold_direction: str = "ge"
    negative_label: str = "low"
    positive_label: str = "high"
    target_column: str | None = None
    time_column: str | None = None
    event_column: str | None = None
    min_class_count: int = 2
    primary_metric: str | None = None
    split_seed: int = 2024
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    cv_val_fraction_of_train: float = 0.2

    @property
    def outcome_column(self) -> str:
        if self.kind == "classification":
            if not self.label_column:
                raise ValueError("classification task requires label_column")
            return self.label_column
        if self.kind == "regression":
            if not self.target_column:
                raise ValueError("regression task requires target_column")
            return self.target_column
        if self.kind in {"prognosis", "survival"}:
            if not self.time_column or not self.event_column:
                raise ValueError("prognosis/survival task requires time_column and event_column")
            return self.time_column
        raise ValueError(f"Unsupported task kind: {self.kind}")

    def validate_for_mil_baseline(self) -> None:
        if self.kind != "classification":
            raise NotImplementedError(
                "The current MIL_BASELINE execution adapter supports classification only. "
                "TaskSpec already records prognosis/survival/regression fields for the next adapters."
            )
        _ = self.outcome_column


@dataclass(frozen=True)
class FeatureSpec:
    format: str = "h5"
    feature_key: str = "features"
    coords_key: str | None = None
    case_id_regex: str = r"^(?P<case>[A-Za-z0-9]+)-"
    feature_glob: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path
    labels_csv: Path
    labels_sheet: str | None = None
    bag_level: str = "case"
    case_id_column: str = "case_id"
    center_column: str | None = None
    cohort_column: str | None = None
    external_test_column: str | None = None
    slide_path_column: str | None = None
    feature: FeatureSpec = field(default_factory=FeatureSpec)


def task_spec_from_config(raw: dict[str, Any]) -> TaskSpec:
    task = dict(raw.get("task", {}))
    kind = str(task.get("kind", task.get("type", "classification"))).lower()
    if kind == "survival":
        kind = "prognosis"
    if kind not in {"classification", "prognosis", "regression"}:
        raise ValueError(f"Unsupported task.kind={kind!r}")
    return TaskSpec(
        kind=kind,  # type: ignore[arg-type]
        label_column=task.get("label_column"),
        label_threshold=float(task["label_threshold"]) if task.get("label_threshold") is not None else None,
        label_threshold_direction=str(task.get("label_threshold_direction", "ge")),
        negative_label=str(task.get("negative_label", "low")),
        positive_label=str(task.get("positive_label", "high")),
        target_column=task.get("target_column"),
        time_column=task.get("time_column"),
        event_column=task.get("event_column"),
        min_class_count=int(task.get("min_class_count", 2)),
        primary_metric=task.get("primary_metric"),
        split_seed=int(task.get("split_seed", 2024)),
        train_size=float(task.get("train_size", 0.7)),
        val_size=float(task.get("val_size", 0.15)),
        test_size=float(task.get("test_size", 0.15)),
        cv_val_fraction_of_train=float(task.get("cv_val_fraction_of_train", 0.2)),
    )


def dataset_spec_from_config(raw: dict[str, Any]) -> DatasetSpec:
    paths = dict(raw.get("paths", {}))
    dataset = dict(raw.get("dataset", {}))
    feature = dict(dataset.get("feature", raw.get("feature", {})))
    return DatasetSpec(
        name=str(dataset.get("name", raw.get("name", "dataset"))),
        data_dir=Path(dataset.get("data_dir", paths.get("data_dir", ""))),
        labels_csv=Path(dataset.get("labels_csv", paths.get("labels_csv", ""))),
        labels_sheet=dataset.get("labels_sheet"),
        bag_level=str(dataset.get("bag_level", "case")).lower(),
        case_id_column=str(dataset.get("case_id_column", "case_id")),
        center_column=dataset.get("center_column"),
        cohort_column=dataset.get("cohort_column"),
        external_test_column=dataset.get("external_test_column"),
        slide_path_column=dataset.get("slide_path_column"),
        feature=FeatureSpec(
            format=str(feature.get("format", "h5")),
            feature_key=str(feature.get("feature_key", "features")),
            coords_key=feature.get("coords_key"),
            case_id_regex=str(feature.get("case_id_regex", r"^(?P<case>[A-Za-z0-9]+)-")),
            feature_glob=feature.get("feature_glob") or feature.get("glob"),
        ),
    )


def specs_to_payload(task: TaskSpec, dataset: DatasetSpec) -> dict[str, Any]:
    return json_ready({"task": asdict(task), "dataset": asdict(dataset)})


def describe_capabilities(task: TaskSpec, dataset: DatasetSpec) -> dict[str, Any]:
    supported_formats = {"h5", "pt"}
    can_execute = task.kind == "classification" and dataset.feature.format.lower() in supported_formats
    return {
        "can_prepare_mil_baseline": can_execute,
        "supported_now": {
            "task_kind": "classification",
            "feature_format": sorted(supported_formats),
            "split_unit": "case/patient",
            "default_bag_level": "case",
        },
        "blocked_reason": None
        if can_execute
        else "Current execution adapter supports classification with H5 or PT feature bags.",
    }
