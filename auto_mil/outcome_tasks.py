from __future__ import annotations

import csv
import ast
import importlib
import inspect
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from .data import (
    DatasetArtifacts,
    KFoldArtifacts,
    _discover_feature_paths,
    _explicit_case_by_path,
    _feature_paths_from_labels,
    _normalize_feature_path,
    _read_coords_array,
    _read_feature_array,
    _read_labels_table,
    case_id_from_h5,
    inspect_feature,
)
from .baseline_registry import assert_mil_baseline_root
from .specs import DatasetSpec, TaskSpec, specs_to_payload
from .state import ExperimentCheckpoint, ResearchJournal, json_ready


@dataclass(frozen=True)
class OutcomeTaskTables:
    slide_df: pd.DataFrame
    case_df: pd.DataFrame
    feature: dict[str, Any]
    missing_cases: Counter
    num_feature_total: int


@dataclass(frozen=True)
class OutcomeRunResult:
    run_id: str
    model_name: str
    fold: int | None
    status: str
    metrics: dict[str, Any]
    output_dir: Path
    error: str | None = None

    @property
    def score(self) -> float:
        if "test_c_index" in self.metrics:
            return _float_or_neg_inf(self.metrics.get("test_c_index"))
        if "test_r2" in self.metrics:
            return _float_or_neg_inf(self.metrics.get("test_r2"))
        rmse = _float_or_neg_inf(self.metrics.get("test_rmse"))
        return -rmse if math.isfinite(rmse) else float("-inf")


def _float_or_neg_inf(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def _outcome_columns(task: TaskSpec) -> list[str]:
    if task.kind == "regression":
        return [task.outcome_column]
    if task.kind == "prognosis":
        return [task.time_column or "", task.event_column or ""]
    raise ValueError(f"Outcome adapter only supports regression/prognosis, got {task.kind!r}")


def _strata(case_df: pd.DataFrame, task: TaskSpec, max_bins: int = 5) -> pd.Series:
    if task.kind == "prognosis":
        return case_df["event"].astype(int).astype(str).radd("event_")
    values = case_df["target"].astype(float)
    bins = min(max_bins, max(2, int(math.sqrt(len(values)))))
    try:
        quantiles = pd.qcut(values, q=bins, duplicates="drop")
        return quantiles.astype(str)
    except ValueError:
        return pd.Series(["all"] * len(case_df), index=case_df.index)


def _can_stratify(labels: pd.Series, n_splits: int = 2) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and not counts.empty and int(counts.min()) >= n_splits


def _safe_holdout_stratify(labels: pd.Series, test_fraction: float) -> pd.Series | None:
    if not _can_stratify(labels):
        return None
    n_test = int(math.ceil(len(labels) * test_fraction))
    n_classes = int(labels.nunique())
    if n_test < n_classes or len(labels) - n_test < n_classes:
        return None
    return labels


def load_outcome_task_tables(dataset: DatasetSpec, task: TaskSpec) -> OutcomeTaskTables:
    if task.kind not in {"regression", "prognosis"}:
        raise ValueError(f"Expected regression or prognosis task, got {task.kind!r}")
    if dataset.feature.format.lower() not in {"h5", "pt"}:
        raise NotImplementedError(f"Unsupported feature format: {dataset.feature.format}")
    if not dataset.labels_csv.exists():
        raise FileNotFoundError(dataset.labels_csv)

    labels = _read_labels_table(dataset.labels_csv, dataset.labels_sheet).copy()
    required = [dataset.case_id_column, *_outcome_columns(task)]
    missing_columns = [column for column in required if column not in labels.columns]
    if missing_columns:
        raise ValueError(f"labels_csv is missing required columns: {missing_columns}")
    labels = labels.dropna(subset=required).copy()
    labels = labels.rename(columns={dataset.case_id_column: "case_id"})
    labels["case_id"] = labels["case_id"].astype(str)
    if task.kind == "regression":
        labels["target"] = pd.to_numeric(labels[task.outcome_column], errors="coerce")
        labels = labels.dropna(subset=["target"])
    else:
        labels["target"] = pd.to_numeric(labels[task.time_column], errors="coerce")
        labels["event"] = pd.to_numeric(labels[task.event_column], errors="coerce")
        labels = labels.dropna(subset=["target", "event"])
        labels = labels.loc[labels["target"] > 0].copy()
        invalid_events = ~labels["event"].isin([0, 1])
        if invalid_events.any():
            values = sorted(labels.loc[invalid_events, "event"].unique().tolist())
            raise ValueError(f"Survival event column must contain only 0/1; got {values}")
        labels["event"] = labels["event"].astype(int)

    outcome_check = ["target", *( ["event"] if task.kind == "prognosis" else [])]
    conflicts = labels.groupby("case_id")[outcome_check].nunique(dropna=False)
    if (conflicts > 1).any(axis=None):
        bad = conflicts[(conflicts > 1).any(axis=1)].index.tolist()[:5]
        raise ValueError(f"Outcome labels are inconsistent within case(s): {bad}")

    optional_columns = [
        dataset.center_column,
        dataset.cohort_column,
        dataset.external_test_column,
        dataset.slide_path_column,
    ]
    keep_columns = list(dict.fromkeys(["case_id", *outcome_check, *[x for x in optional_columns if x and x in labels.columns]]))
    labels = labels[keep_columns].drop_duplicates("case_id").copy()
    label_map = labels.set_index("case_id").to_dict("index")
    feature_paths = _feature_paths_from_labels(labels, dataset) if dataset.slide_path_column else _discover_feature_paths(dataset)
    if not feature_paths:
        raise FileNotFoundError(f"No {dataset.feature.format.upper()} feature files found for {dataset.name}")
    explicit_by_path = _explicit_case_by_path(labels, dataset) if dataset.slide_path_column else {}
    rows: list[dict[str, Any]] = []
    missing_cases: Counter = Counter()
    for path in feature_paths:
        case_id = explicit_by_path.get(str(path)) or case_id_from_h5(path, dataset.feature.case_id_regex)
        item = label_map.get(case_id)
        if item is None:
            missing_cases[case_id] += 1
            continue
        row: dict[str, Any] = {"case_id": case_id, "slide_path": str(path), "target": float(item["target"])}
        if task.kind == "prognosis":
            row["event"] = int(item["event"])
        for column in (dataset.center_column, dataset.cohort_column, dataset.external_test_column):
            if column and column in item:
                row[column] = item[column]
        rows.append(row)
    slide_df = pd.DataFrame(rows)
    if slide_df.empty:
        raise ValueError("No feature bags matched the outcome table")
    case_columns = ["case_id", "target", *( ["event"] if task.kind == "prognosis" else [])]
    case_columns.extend(column for column in (dataset.center_column, dataset.cohort_column, dataset.external_test_column) if column and column in slide_df.columns)
    case_df = slide_df[case_columns].drop_duplicates("case_id").reset_index(drop=True)
    case_df["stratum"] = _strata(case_df, task)
    case_df["label_name"] = case_df["stratum"]
    slide_df = slide_df.merge(case_df[["case_id", "stratum", "label_name"]], on="case_id", how="left")
    feature = inspect_feature(Path(slide_df.iloc[0]["slide_path"]), dataset)
    return OutcomeTaskTables(slide_df, case_df, feature, missing_cases, len(feature_paths))


def _case_bags(slide_df: pd.DataFrame, output_dir: Path, dataset: DatasetSpec, task: TaskSpec) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if dataset.bag_level == "slide":
        return slide_df.copy(), None
    if dataset.bag_level != "case":
        raise ValueError("dataset.bag_level must be 'case' or 'slide'")
    bag_dir = output_dir / "case_bags"
    bag_dir.mkdir(parents=True, exist_ok=True)
    bag_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for case_id, group in slide_df.sort_values("slide_path").groupby("case_id", sort=True):
        if group["split"].nunique() != 1:
            raise ValueError(f"Case {case_id} appears in multiple splits")
        features = []
        coords = []
        coords_complete = True
        source_paths = [Path(value) for value in group["slide_path"].astype(str)]
        for path in source_paths:
            item = _read_feature_array(path, dataset)
            features.append(item)
            item_coords = _read_coords_array(path, dataset)
            if item_coords is None or item_coords.shape[0] != item.shape[0]:
                coords_complete = False
            else:
                coords.append(item_coords)
            manifest_rows.append({"case_id": case_id, "source_slide_path": str(path), "n_patches": int(item.shape[0])})
        bag_path = bag_dir / f"{str(case_id).replace('/', '_').replace('\\', '_')}.h5"
        with h5py.File(bag_path, "w") as h5:
            h5.create_dataset("features", data=np.concatenate(features, axis=0))
            if coords_complete and coords:
                h5.create_dataset("coords", data=np.concatenate(coords, axis=0))
            h5.attrs["case_id"] = str(case_id)
        first = group.iloc[0].to_dict()
        first["slide_path"] = str(bag_path)
        first["n_source_slides"] = len(source_paths)
        first["n_patches"] = int(sum(item.shape[0] for item in features))
        bag_rows.append(first)
    return pd.DataFrame(bag_rows), pd.DataFrame(manifest_rows)


def _metadata(dataset: DatasetSpec, task: TaskSpec, tables: OutcomeTaskTables, bag_df: pd.DataFrame) -> dict[str, Any]:
    payload = specs_to_payload(task, dataset)
    metadata: dict[str, Any] = {
        "dataset": dataset.name,
        "task": payload["task"],
        "dataset_spec": payload["dataset"],
        "num_cases": int(tables.case_df.shape[0]),
        "num_feature_total": tables.num_feature_total,
        "num_feature_matched": int(tables.slide_df.shape[0]),
        "num_training_bags": int(bag_df.shape[0]),
        "feature": tables.feature,
        "missing_label_cases": dict(tables.missing_cases),
        "bag_level": dataset.bag_level,
        "outcome": {
            "column": task.outcome_column,
            "mean": float(tables.case_df["target"].mean()),
            "std": float(tables.case_df["target"].std(ddof=0)),
            "min": float(tables.case_df["target"].min()),
            "max": float(tables.case_df["target"].max()),
        },
    }
    if task.kind == "prognosis":
        metadata["survival"] = {
            "time_column": task.time_column,
            "event_column": task.event_column,
            "events": int(tables.case_df["event"].sum()),
            "censored": int((1 - tables.case_df["event"]).sum()),
        }
    return metadata


def _write_fold(
    *,
    tables: OutcomeTaskTables,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: Path,
    split_by_case: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
) -> DatasetArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df = tables.slide_df.copy()
    slide_df["split"] = slide_df["case_id"].astype(str).map(split_by_case)
    slide_df = slide_df.dropna(subset=["split"]).copy()
    bag_df, manifest = _case_bags(slide_df, output_dir, dataset, task)
    bag_csv = output_dir / "bags.csv"
    bag_df.to_csv(bag_csv, index=False)
    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
    if manifest is not None:
        manifest.to_csv(output_dir / "case_bag_manifest.csv", index=False)
    metadata = _metadata(dataset, task, tables, bag_df)
    metadata["split_counts_cases"] = {
        str(key): int(value) for key, value in bag_df.groupby("split")["case_id"].nunique().items()
    }
    metadata["stratum_counts"] = {
        str(key): int(value) for key, value in bag_df.groupby(["split", "stratum"]).size().items()
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    paths_path = output_dir / "feature_paths.csv"
    pd.DataFrame({"feature_path": bag_df["slide_path"].astype(str)}).to_csv(paths_path, index=False)
    return DatasetArtifacts(dataset_csv=bag_csv, h5_paths_csv=paths_path, metadata_json=metadata_path)


def _split_train_val(case_df: pd.DataFrame, task: TaskSpec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(case_df) < 3:
        raise ValueError("At least three development cases are required for a train/validation split")
    labels = case_df["stratum"]
    stratify = labels if _can_stratify(labels) else None
    test_size = min(max(task.cv_val_fraction_of_train, 0.1), 0.5)
    n_val = int(math.ceil(len(case_df) * test_size))
    n_classes = int(labels.nunique())
    if stratify is not None and (n_val < n_classes or len(case_df) - n_val < n_classes):
        stratify = None
    return train_test_split(case_df, test_size=test_size, random_state=seed, stratify=stratify)


def prepare_outcome_dataset(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    split_by_case: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DatasetArtifacts:
    tables = load_outcome_task_tables(dataset, task)
    if split_by_case is None:
        case_df = tables.case_df
        labels = case_df["stratum"]
        holdout_fraction = task.test_size
        stratify = _safe_holdout_stratify(labels, holdout_fraction)
        trainval, test = train_test_split(
            case_df,
            train_size=task.train_size + task.val_size,
            random_state=task.split_seed,
            stratify=stratify,
        )
        train, val = _split_train_val(trainval, task, task.split_seed)
        split_by_case = {
            **{str(value): "train" for value in train["case_id"]},
            **{str(value): "val" for value in val["case_id"]},
            **{str(value): "test" for value in test["case_id"]},
        }
    return _write_fold(
        tables=tables,
        dataset=dataset,
        task=task,
        output_dir=Path(output_dir),
        split_by_case=split_by_case,
        extra_metadata=extra_metadata,
    )


def prepare_outcome_kfold(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    n_splits: int = 5,
    extra_metadata: dict[str, Any] | None = None,
) -> KFoldArtifacts:
    tables = load_outcome_task_tables(dataset, task)
    case_df = tables.case_df.reset_index(drop=True)
    labels = case_df["stratum"]
    if _can_stratify(labels, n_splits):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=task.split_seed)
        iterator = splitter.split(case_df, labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=task.split_seed)
        iterator = splitter.split(case_df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs: list[Path] = []
    for fold, (trainval_idx, test_idx) in enumerate(iterator):
        trainval = case_df.iloc[trainval_idx]
        test = case_df.iloc[test_idx]
        train, val = _split_train_val(trainval, task, task.split_seed + fold)
        split_by_case = {
            **{str(value): "train" for value in train["case_id"]},
            **{str(value): "val" for value in val["case_id"]},
            **{str(value): "test" for value in test["case_id"]},
        }
        fold_dir = output_dir / f"fold_{fold}"
        _write_fold(
            tables=tables,
            dataset=dataset,
            task=task,
            output_dir=fold_dir,
            split_by_case=split_by_case,
            extra_metadata=extra_metadata,
        )
        fold_dirs.append(fold_dir)
    root_metadata = _metadata(dataset, task, tables, tables.slide_df)
    root_metadata.update({"n_splits": n_splits, "stratified": isinstance(splitter, StratifiedKFold), "fold_dirs": [str(path) for path in fold_dirs]})
    if extra_metadata:
        root_metadata.update(extra_metadata)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(json_ready(root_metadata), indent=2), encoding="utf-8")
    return KFoldArtifacts(fold_dirs=fold_dirs, metadata_json=metadata_path)


class OutcomeBagDataset(Dataset):
    def __init__(self, bag_csv: str | Path, split: str, dataset: DatasetSpec, max_patches: int | None, seed: int):
        self.frame = pd.read_csv(bag_csv)
        self.frame = self.frame.loc[self.frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No bags in split={split!r} for {bag_csv}")
        self.dataset = dataset
        self.max_patches = max_patches
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        path = Path(str(row["slide_path"]))
        if path.suffix.lower() == ".h5":
            with h5py.File(path, "r") as h5:
                key = "features" if "features" in h5 else self.dataset.feature.feature_key
                feature = torch.from_numpy(np.asarray(h5[key]))
                coords_key = self.dataset.feature.coords_key or ("coords" if "coords" in h5 else None)
                coords = torch.from_numpy(np.asarray(h5[coords_key])) if coords_key and coords_key in h5 else None
        else:
            feature = torch.from_numpy(_read_feature_array(path, self.dataset))
            array_coords = _read_coords_array(path, self.dataset)
            coords = torch.from_numpy(array_coords) if array_coords is not None else None
        if feature.ndim == 3:
            feature = feature.reshape(-1, feature.shape[-1])
        if coords is not None and coords.ndim == 3:
            coords = coords.reshape(-1, coords.shape[-1])
        if coords is not None and (coords.ndim != 2 or coords.shape[0] != feature.shape[0]):
            coords = None
        if self.max_patches and feature.shape[0] > self.max_patches:
            generator = torch.Generator().manual_seed(self.seed + index)
            keep = torch.randperm(feature.shape[0], generator=generator)[: self.max_patches]
            keep = keep.sort().values
            feature = feature.index_select(0, keep)
            if coords is not None:
                coords = coords.index_select(0, keep)
        event = float(row["event"]) if "event" in row.index and not pd.isna(row["event"]) else float("nan")
        safe_coords = coords.float() if coords is not None else torch.empty((0, 2), dtype=torch.float32)
        return feature.float(), torch.tensor(float(row["target"])), torch.tensor(event), safe_coords, str(row["case_id"])


VENDORED_OUTCOME_MODELS = {
    "AB_MIL", "AC_MIL", "ADD_MIL", "AEM_MIL", "AMD_MIL", "CA_MIL", "CDP_MIL", "DAG_MIL", "DG_MIL",
    "DGR_MIL", "DT_MIL", "DYHG_MIL", "FOURIER_MIL", "FR_MIL", "GATE_AB_MIL", "GDF_MIL", "IB_MIL",
    "IIB_MIL", "ILRA_MIL", "LONG_MIL", "MAMBA_MIL", "MAMBA2D_MIL", "MAX_MIL", "MEAN_MIL", "MHIM_MIL",
    "MICO_MIL", "MICRO_MIL", "MO_MIL", "MSM_MIL", "NCIE_MIL", "PA_MIL", "PGCN_MIL", "PSA_MIL",
    "RET_MIL", "RRT_MIL", "S4_MIL", "SC_MIL", "STABLE_MIL", "TDA_MIL", "TRANS_MIL", "WIKG_MIL",
}


def _normalized_model_name(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _activation(value: Any) -> nn.Module:
    if isinstance(value, nn.Module):
        return value
    name = str(value or "relu").lower()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def _resolve_vendored_model_class(root: Path, model_name: str) -> type[nn.Module]:
    model_dir = root / "modules" / model_name
    target = _normalized_model_name(model_name)
    matches: list[tuple[Path, str]] = []
    for source_path in model_dir.glob("*.py"):
        if source_path.name == "__init__.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8", errors="ignore"), filename=str(source_path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _normalized_model_name(node.name) == target:
                matches.append((source_path, node.name))
    exact = [item for item in matches if item[1] == model_name]
    if len(exact) == 1:
        matches = exact
    if len(matches) != 1:
        raise ValueError(f"Could not resolve one model class for {model_name}: {matches}")
    source_path, class_name = matches[0]
    module_name = f"modules.{model_name}.{source_path.stem}"
    candidate = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(candidate, nn.Module):
        raise TypeError(f"{module_name}.{class_name} is not a torch module")
    return candidate


def _generic_vendored_outcome_backbone(
    model_name: str,
    *,
    in_dim: int,
    dropout: float,
    root: Path,
    overrides: dict[str, Any],
) -> nn.Module:
    config_path = root / "configs" / f"{model_name}.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config_params = dict(payload.get("Model", {}))
    config_params.update(overrides)
    if model_name == "RET_MIL":
        requested_heads = int(config_params.get("num_heads", 8))
        valid_heads = [head for head in range(min(requested_heads, in_dim), 0, -1) if in_dim % head == 0 and (in_dim // head) % 2 == 0]
        if not valid_heads:
            raise ValueError(f"RET_MIL requires an input dimension divisible into even attention heads; got in_dim={in_dim}")
        config_params["num_heads"] = valid_heads[0]
    if model_name == "NCIE_MIL":
        requested_heads = int(config_params.get("num_heads", 4))
        latent_dim = int(config_params.get("latent_dim", 1024))
        valid_heads = [head for head in range(min(requested_heads, in_dim), 0, -1) if in_dim % head == 0 and latent_dim % head == 0]
        if not valid_heads:
            raise ValueError(f"NCIE_MIL requires a shared divisor of in_dim={in_dim} and latent_dim={latent_dim}")
        config_params["num_heads"] = valid_heads[0]
    cls = _resolve_vendored_model_class(root, model_name)
    parameters: dict[str, Any] = {}
    for parameter in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        key = parameter.name
        if key in {"num_classes", "n_classes", "n_class"}:
            # CDP uses num_classes as the number of variational mixture
            # components rather than a terminal classification head.
            parameters[key] = int(config_params.get("num_components", 8)) if model_name == "CDP_MIL" else 1
        elif key == "in_dim":
            parameters[key] = in_dim
        elif key == "dropout":
            parameters[key] = dropout
        elif key == "act":
            parameters[key] = _activation(config_params.get(key, "relu"))
        elif key in config_params and not isinstance(config_params[key], dict):
            parameters[key] = config_params[key]
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"{model_name} requires unsupported constructor parameter {key!r}")
    model = cls(**parameters)
    if model_name == "WIKG_MIL" and not hasattr(model, "message_dropout"):
        # The vendored implementation omits this attribute when dropout=0,
        # although forward always invokes it. Preserve zero-dropout semantics.
        model.message_dropout = nn.Identity()
    if model_name == "RET_MIL":
        # RET_MIL does not expose its bag feature. Replacing this inner head
        # makes the wrapper's logits key carry RetMIL's global bag embedding.
        model.retmil.classifier = nn.Identity()
    return model


def _vendored_outcome_backbone(
    model_name: str,
    *,
    in_dim: int,
    dropout: float,
    mil_baseline_dir: str | Path | None,
    overrides: dict[str, Any] | None = None,
) -> tuple[nn.Module, int | None]:
    """Reuse a vendored MIL aggregator while replacing its classifier downstream."""
    name = model_name.upper()
    root = assert_mil_baseline_root(mil_baseline_dir)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    params = dict(overrides or {})
    if name == "AB_MIL":
        cls = importlib.import_module("modules.AB_MIL.ab_mil").AB_MIL
        hidden_dim = int(params.pop("L", 256))
        model = cls(L=hidden_dim, D=int(params.pop("D", 128)), num_classes=1, dropout=dropout, in_dim=in_dim, **params)
        model.classifier = nn.Identity()
        return model, hidden_dim
    if name == "TRANS_MIL":
        cls = importlib.import_module("modules.TRANS_MIL.trans_mil").TRANS_MIL
        hidden_dim = int(params.pop("hidden_dim", 256))
        model = cls(
            num_classes=1,
            dropout=dropout,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=int(params.pop("num_heads", 8)),
            **params,
        )
        model._fc2 = nn.Identity()
        return model, hidden_dim
    if name == "RRT_MIL":
        cls = importlib.import_module("modules.RRT_MIL.rrt_mil").RRT_MIL
        hidden_dim = int(params.pop("L", 256))
        model = cls(L=hidden_dim, D=int(params.pop("D", 128)), num_classes=1, dropout=dropout, in_dim=in_dim, **params)
        model.classifier = nn.Identity()
        return model, hidden_dim
    if name == "STABLE_MIL":
        cls = importlib.import_module("modules.STABLE_MIL.stable_mil").STABLE_MIL
        hidden_dim = int(params.pop("hidden_dim", 256))
        model = cls(
            num_classes=1,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            depth=int(params.pop("depth", 2)),
            num_heads=int(params.pop("num_heads", 8)),
            aggregate_num=int(params.pop("aggregate_num", 64)),
            k_neighbors=int(params.pop("k_neighbors", 4)),
            dropout=dropout,
            **params,
        )
        model.head = nn.Identity()
        return model, hidden_dim
    if name == "GDF_MIL":
        cls = importlib.import_module("modules.GDF_MIL.gdf_mil").GDF_MIL
        hidden_dim = int(params.pop("hid_dim", 256))
        out_dim = int(params.pop("out_dim", 128))
        k_components = int(params.pop("k_components", 5))
        model = cls(
            in_dim=in_dim,
            num_classes=1,
            hid_dim=hidden_dim,
            out_dim=out_dim,
            k_components=k_components,
            k_neighbors=int(params.pop("k_neighbors", min(4, k_components))),
            dropout=dropout,
            **params,
        )
        model.classifier = nn.Identity()
        return model, out_dim
    if name not in VENDORED_OUTCOME_MODELS:
        raise ValueError(f"No vendored outcome backbone is registered for {model_name!r}")
    return _generic_vendored_outcome_backbone(
        name,
        in_dim=in_dim,
        dropout=dropout,
        root=root,
        overrides=params,
    ), None


class OutcomeMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_name: str,
        dropout: float = 0.1,
        mil_baseline_dir: str | Path | None = None,
        model_overrides: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model_name = model_name.upper()
        self.backbone: nn.Module | None = None
        self.task_head: nn.Module | None = None
        self.backbone_accepts_coords = False
        self.expected_instances: int | None = None
        if self.model_name in VENDORED_OUTCOME_MODELS:
            self.backbone, feature_dim = _vendored_outcome_backbone(
                self.model_name,
                in_dim=in_dim,
                dropout=dropout,
                mil_baseline_dir=mil_baseline_dir,
                overrides=model_overrides,
            )
            self.task_head = nn.Linear(feature_dim, 1) if feature_dim is not None else nn.LazyLinear(1)
            self.backbone_accepts_coords = "coords" in inspect.signature(self.backbone.forward).parameters
            self.expected_instances = int(self.backbone.in_chans) if self.model_name == "NCIE_MIL" else None
            return
        self.encoder = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.attention_w = nn.Linear(128, 1)
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        if self.backbone is not None:
            if self.expected_instances is not None:
                x, coords = _resize_bag_instances(x, coords, self.expected_instances)
            if self.model_name == "NCIE_MIL":
                # NCIE pools each single WSI into a 1x1 feature map. Keep its
                # BatchNorm layers in inference mode for Auto-MIL's batch=1 bags.
                for module in self.backbone.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            kwargs: dict[str, Any] = {"return_WSI_feature": True}
            if self.backbone_accepts_coords and coords is not None and coords.numel() > 0:
                kwargs["coords"] = coords
            output = self.backbone(x, **kwargs)
            feature = output.get("WSI_feature")
            if feature is None:
                feature = output.get("logits")
            if feature is None:
                raise ValueError(f"{self.model_name} did not expose a bag-level feature")
            if feature.ndim == 1:
                feature = feature.unsqueeze(0)
            if self.task_head is None:
                raise RuntimeError("Vendored outcome backbone is missing its task head")
            return self.task_head(feature.float()).reshape(-1)
        if x.ndim == 3:
            x = x.squeeze(0)
        feature = self.encoder(x)
        if self.model_name == "MEAN_MIL":
            bag = feature.mean(dim=0, keepdim=True)
        elif self.model_name == "MAX_MIL":
            bag = feature.max(dim=0, keepdim=True).values
        elif self.model_name in {"AB_MIL", "GATE_AB_MIL"}:
            attention_feature = self.attention_v(feature)
            if self.model_name == "GATE_AB_MIL":
                attention_feature = attention_feature * self.attention_u(feature)
            weights = torch.softmax(self.attention_w(attention_feature).transpose(0, 1), dim=1)
            bag = weights @ feature
        else:
            raise ValueError(
                f"Outcome adapter supports {len(VENDORED_OUTCOME_MODELS)} audited vendored head/loss backbones; "
                f"got {self.model_name}"
            )
        return self.head(bag).reshape(-1)


def _resize_bag_instances(
    features: torch.Tensor,
    coords: torch.Tensor | None,
    expected_instances: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pad or evenly subsample models that require a fixed square token grid."""
    if features.shape[1] == expected_instances:
        return features, coords
    source = features.shape[1]
    if source <= 0:
        raise ValueError("Outcome bag has no patch features")
    if source > expected_instances:
        indices = torch.linspace(0, source - 1, expected_instances, device=features.device).round().long()
    else:
        indices = torch.arange(expected_instances, device=features.device) % source
    resized_features = features.index_select(1, indices)
    resized_coords = coords.index_select(1, indices) if coords is not None and coords.numel() > 0 else coords
    return resized_features, resized_coords


def _cox_partial_loss(risk: torch.Tensor, time_values: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time_values, descending=True)
    ordered_risk = risk[order]
    ordered_event = events[order] > 0.5
    if int(ordered_event.sum()) == 0:
        raise ValueError("Survival training split has no observed events")
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    return -(ordered_risk[ordered_event] - log_risk_set[ordered_event]).mean()


def concordance_index(time_values: list[float], events: list[int], risk_scores: list[float]) -> float:
    permissible = 0
    concordant = 0.0
    for i, (time_i, event_i, risk_i) in enumerate(zip(time_values, events, risk_scores)):
        if int(event_i) != 1:
            continue
        for j, (time_j, risk_j) in enumerate(zip(time_values, risk_scores)):
            if i == j or time_i >= time_j:
                continue
            permissible += 1
            if risk_i > risk_j:
                concordant += 1.0
            elif risk_i == risk_j:
                concordant += 0.5
    return concordant / permissible if permissible else float("nan")


def _metrics(task: TaskSpec, targets: list[float], events: list[int], predictions: list[float]) -> dict[str, float]:
    if task.kind == "prognosis":
        return {"c_index": concordance_index(targets, events, predictions)}
    target = np.asarray(targets, dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    spearman = pd.Series(target).corr(pd.Series(prediction), method="spearman") if len(target) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)) if len(target) >= 2 else float("nan"),
        "spearman": float(spearman) if spearman is not None else float("nan"),
    }


def _evaluate(model: nn.Module, loader: DataLoader, task: TaskSpec, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    targets: list[float] = []
    events: list[int] = []
    predictions: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for features, target, event, coords, case_id in loader:
            prediction = float(model(features.to(device), coords.to(device)).detach().cpu().item())
            target_value = float(target.item())
            event_value = int(event.item()) if not torch.isnan(event).item() else 0
            targets.append(target_value)
            events.append(event_value)
            predictions.append(prediction)
            row = {"case_id": str(case_id[0]), "target": target_value, "y_pred": prediction}
            if task.kind == "prognosis":
                row["event"] = event_value
            rows.append(row)
    return _metrics(task, targets, events, predictions), rows


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_outcome_mil(
    *,
    task: TaskSpec,
    dataset: DatasetSpec,
    bag_csv: Path,
    output_dir: Path,
    model_name: str,
    in_dim: int,
    epochs: int,
    lr: float,
    dropout: float,
    device_id: int,
    max_patches: int | None,
    seed: int,
    mil_baseline_dir: str | Path | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    train_set = OutcomeBagDataset(bag_csv, "train", dataset, max_patches, seed)
    val_set = OutcomeBagDataset(bag_csv, "val", dataset, max_patches, seed + 10_000)
    test_set = OutcomeBagDataset(bag_csv, "test", dataset, max_patches, seed + 20_000)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model = OutcomeMIL(
        in_dim,
        model_name,
        dropout,
        mil_baseline_dir=mil_baseline_dir,
        model_overrides=model_overrides,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    best_value = float("inf") if task.kind == "regression" else float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        if task.kind == "prognosis":
            risks: list[torch.Tensor] = []
            times: list[torch.Tensor] = []
            events: list[torch.Tensor] = []
            for features, target, event, coords, _case_id in train_loader:
                risks.append(model(features.to(device), coords.to(device)))
                times.append(target.to(device).reshape(-1))
                events.append(event.to(device).reshape(-1))
            optimizer.zero_grad()
            loss = _cox_partial_loss(torch.cat(risks), torch.cat(times), torch.cat(events))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        else:
            for features, target, _event, coords, _case_id in train_loader:
                optimizer.zero_grad()
                prediction = model(features.to(device), coords.to(device))
                loss = F.mse_loss(prediction, target.to(device).reshape(-1))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        val_metrics, _ = _evaluate(model, val_loader, task, device)
        test_metrics, _ = _evaluate(model, test_loader, task, device)
        candidate = val_metrics["rmse"] if task.kind == "regression" else val_metrics["c_index"]
        improved = math.isfinite(candidate) and (candidate < best_value if task.kind == "regression" else candidate > best_value)
        if improved:
            best_value = candidate
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row: dict[str, Any] = {"epoch": epoch, "train_loss": float(mean(losses)), **{f"val_{key}": value for key, value in val_metrics.items()}, **{f"test_{key}": value for key, value in test_metrics.items()}}
        history.append(row)
    if best_state is None:
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_model.pth")
    final_metrics: dict[str, Any] = {}
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        split_metrics, prediction_rows = _evaluate(model, loader, task, device)
        final_metrics.update({f"{split}_{key}": value for key, value in split_metrics.items()})
        pd.DataFrame(prediction_rows).to_csv(output_dir / f"{split}_predictions.csv", index=False)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final_metrics["metrics_path"] = str(output_dir / "metrics.csv")
    final_metrics["device"] = str(device)
    return final_metrics


def _run_task_cv_model(
    *,
    cfg: Any,
    task: TaskSpec,
    dataset: DatasetSpec,
    fold_dir: Path,
    model_name: str,
    fold: int | None,
    output_dir: Path,
    epochs: int,
    seed: int,
) -> OutcomeRunResult:
    metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    training = cfg.raw.get("training", {})
    search = cfg.raw.get("search", {})
    outcome_models = cfg.raw.get("outcome_models", {})
    model_overrides = outcome_models.get(model_name, outcome_models.get(model_name.upper(), {}))
    if not isinstance(model_overrides, dict):
        raise ValueError("outcome_models entries must be mappings of constructor overrides")
    max_patches = training.get("max_patches_per_bag")
    max_patches = int(max_patches) if max_patches not in (None, "", 0) else None
    run_id = f"{model_name.lower()}_fold{fold}" if fold is not None else model_name.lower()
    run_dir = output_dir / "runs" / run_id
    try:
        metrics = train_outcome_mil(
            task=task,
            dataset=dataset,
            bag_csv=fold_dir / "bags.csv",
            output_dir=run_dir,
            model_name=model_name,
            in_dim=int(metadata["feature"]["in_dim"]),
            epochs=epochs,
            lr=float(search.get("learning_rates", [0.0002])[0]),
            dropout=float(search.get("dropouts", [0.1])[0]),
            device_id=int(training.get("device", 0)),
            max_patches=max_patches,
            seed=seed,
            mil_baseline_dir=cfg.mil_baseline_dir,
            model_overrides=model_overrides,
        )
        return OutcomeRunResult(run_id, model_name, fold, "completed", metrics, run_dir)
    except Exception as exc:  # noqa: BLE001 - preserve failed experiments as evidence.
        return OutcomeRunResult(run_id, model_name, fold, "failed", {}, run_dir, str(exc))


def _summary(results: list[OutcomeRunResult], task: TaskSpec) -> dict[str, dict[str, Any]]:
    metric_names = ["test_c_index", "val_c_index"] if task.kind == "prognosis" else ["test_mae", "test_rmse", "test_r2", "test_spearman", "val_rmse"]
    summary: dict[str, dict[str, Any]] = {}
    for model in sorted({item.model_name for item in results}):
        rows = [item for item in results if item.model_name == model]
        payload: dict[str, Any] = {"n_completed": sum(item.status == "completed" for item in rows), "n_total": len(rows)}
        for metric in metric_names:
            values = [float(item.metrics[metric]) for item in rows if metric in item.metrics and math.isfinite(float(item.metrics[metric]))]
            if values:
                payload[f"{metric}_mean"] = mean(values)
                payload[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary[model] = payload
    return summary


def _write_task_report(output_dir: Path, task: TaskSpec, results: list[OutcomeRunResult], summary: dict[str, dict[str, Any]]) -> Path:
    primary = "test_c_index" if task.kind == "prognosis" else "test_rmse"
    reverse = task.kind == "prognosis"
    ranked = sorted(summary.items(), key=lambda item: item[1].get(f"{primary}_mean", float("-inf") if reverse else float("inf")), reverse=reverse)
    lines = ["# Outcome MIL CV Report", "", f"- Task: `{task.kind}`", f"- Primary metric: `{primary}`", "", "| Rank | Model | Completed | Primary metric | Secondary metrics |", "|---:|---|---:|---:|---|"]
    for rank, (model, row) in enumerate(ranked, start=1):
        value = row.get(f"{primary}_mean")
        spread = row.get(f"{primary}_std")
        text = "" if value is None else f"{value:.4f} +/- {spread:.4f}"
        secondary = ", ".join(f"{key}={value:.4f}" for key, value in row.items() if key.endswith("_mean") and key != f"{primary}_mean")
        lines.append(f"| {rank} | {model} | {row['n_completed']}/{row['n_total']} | {text} | {secondary} |")
    lines.extend(["", "## Per-Fold Runs", "", "| Run | Model | Fold | Status | Metrics | Error |", "|---|---|---:|---|---|---|"])
    for result in results:
        lines.append(f"| `{result.run_id}` | {result.model_name} | {result.fold if result.fold is not None else ''} | {result.status} | `{result.metrics}` | {result.error or ''} |")
    path = output_dir / "cv_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_outcome_cv(
    cfg: Any,
    *,
    models: list[str] | None,
    n_splits: int,
    epochs: int | None,
    split_plan_path: Path | None = None,
    split_plan_id: str | None = None,
    resume: bool = False,
) -> Path:
    task = cfg.task_spec
    dataset = cfg.dataset_spec
    if task.kind not in {"regression", "prognosis"}:
        raise ValueError("run_outcome_cv requires regression or prognosis")
    if split_plan_path is not None:
        from .split_executor import select_split_plan

        selected = select_split_plan(split_plan_path, split_plan_id)
        if selected.strategy != "n_fold_cross_validation":
            raise ValueError("run-cv requires an n_fold_cross_validation plan")
        n_splits = int(selected.plan["n_splits"])
    else:
        selected = None
    output_dir = cfg.output_dir / "case_level_cv"
    confirmed_split = (
        {"confirmed_split": {"split_plan_path": str(split_plan_path), "plan_id": selected.plan_id, "plan": selected.plan}}
        if selected
        else None
    )
    artifacts = prepare_outcome_kfold(
        dataset=dataset,
        task=task,
        output_dir=output_dir / "folds",
        n_splits=n_splits,
        extra_metadata=confirmed_split,
    )
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    selected_models = models or list(cfg.raw.get("search", {}).get("cv_models", ["MEAN_MIL", "MAX_MIL", "AB_MIL"]))
    recipe_epochs = int(epochs if epochs is not None else cfg.raw.get("training", {}).get("cv_epochs", 3))
    results: list[OutcomeRunResult] = []
    for fold, fold_dir in enumerate(artifacts.fold_dirs):
        for model_name in selected_models:
            run_id = f"{model_name.lower()}_fold{fold}"
            cached = checkpoint.get_completed_payload(run_id) if resume else None
            if cached:
                results.append(OutcomeRunResult(run_id, model_name, fold, str(cached["status"]), dict(cached.get("metrics", {})), Path(cached.get("output_dir", output_dir)), cached.get("error")))
                continue
            journal.write("outcome_cv_run_start", {"run_id": run_id, "task": task.kind, "fold": fold, "model_name": model_name})
            result = _run_task_cv_model(cfg=cfg, task=task, dataset=dataset, fold_dir=fold_dir, model_name=model_name, fold=fold, output_dir=output_dir, epochs=recipe_epochs, seed=task.split_seed + fold)
            results.append(result)
            payload = json_ready(asdict(result))
            payload["recipe"] = {"recipe_id": run_id, "model_name": model_name}
            checkpoint.record_run(run_id, "outcome_cv", result.status, payload)
            journal.write("outcome_cv_run_result", payload)
    summary = _summary(results, task)
    summary_path = output_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _write_task_report(output_dir, task, results, summary)
    checkpoint.update_metadata(
        command="run-cv",
        task_kind=task.kind,
        n_splits=n_splits,
        metadata_json=str(artifacts.metadata_json),
        report=str(report),
        summary_json=str(summary_path),
        confirmed_split=confirmed_split["confirmed_split"] if confirmed_split else None,
    )
    return report


def run_outcome_holdout(
    cfg: Any,
    *,
    models: list[str] | None = None,
    epochs: int | None = None,
    split_plan_path: Path | None = None,
    split_plan_id: str | None = None,
    resume: bool = False,
) -> Path:
    task = cfg.task_spec
    dataset = cfg.dataset_spec
    output_dir = cfg.output_dir
    if split_plan_path is not None:
        from .split_executor import select_split_plan

        selected = select_split_plan(split_plan_path, split_plan_id)
        split_by_case = _outcome_holdout_case_split(load_outcome_task_tables(dataset, task).case_df, dataset, task, selected)
        extra_metadata = {
            "confirmed_split": {
                "split_plan_path": str(split_plan_path),
                "plan_id": selected.plan_id,
                "plan": selected.plan,
            }
        }
    else:
        selected = None
        split_by_case = None
        extra_metadata = None
    artifacts = prepare_outcome_dataset(
        dataset=dataset,
        task=task,
        output_dir=output_dir / "dataset",
        split_by_case=split_by_case,
        extra_metadata=extra_metadata,
    )
    selected_models = models or list(cfg.raw.get("search", {}).get("screen_models", ["MEAN_MIL", "MAX_MIL", "AB_MIL"]))
    run_epochs = int(epochs if epochs is not None else cfg.raw.get("training", {}).get("screening_epochs", 3))
    checkpoint = ExperimentCheckpoint(output_dir / "checkpoint.json")
    journal = ResearchJournal(output_dir / "research_journal.jsonl")
    results: list[OutcomeRunResult] = []
    for model_name in selected_models:
        run_id = model_name.lower()
        cached = checkpoint.get_completed_payload(run_id) if resume else None
        if cached:
            results.append(OutcomeRunResult(run_id, model_name, None, str(cached["status"]), dict(cached.get("metrics", {})), Path(cached.get("output_dir", output_dir)), cached.get("error")))
            continue
        journal.write("outcome_holdout_run_start", {"run_id": run_id, "task": task.kind, "model_name": model_name})
        result = _run_task_cv_model(cfg=cfg, task=task, dataset=dataset, fold_dir=artifacts.dataset_csv.parent, model_name=model_name, fold=None, output_dir=output_dir, epochs=run_epochs, seed=task.split_seed)
        results.append(result)
        payload = json_ready(asdict(result))
        payload["recipe"] = {"recipe_id": run_id, "model_name": model_name}
        checkpoint.record_run(run_id, "outcome_holdout", result.status, payload)
        journal.write("outcome_holdout_run_result", payload)
    summary = _summary(results, task)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _write_task_report(output_dir, task, results, summary)
    checkpoint.update_metadata(
        command="run",
        task_kind=task.kind,
        metadata_json=str(artifacts.metadata_json),
        report=str(report),
        summary_json=str(summary_path),
        confirmed_split=extra_metadata["confirmed_split"] if extra_metadata else None,
    )
    return report


def _outcome_holdout_case_split(case_df: pd.DataFrame, dataset: DatasetSpec, task: TaskSpec, selected: Any) -> dict[str, str]:
    if selected.strategy == "external_test":
        column = dataset.external_test_column
        values = {str(value) for value in selected.plan.get("external_test_values", [])}
        if not column or column not in case_df.columns or not values:
            raise ValueError("External-test plan requires dataset.external_test_column and external_test_values.")
        test = case_df.loc[case_df[column].astype(str).isin(values)].copy()
        trainval = case_df.loc[~case_df["case_id"].astype(str).isin(test["case_id"].astype(str))].copy()
    elif selected.strategy == "center_external_test":
        column = dataset.center_column
        centers = {str(value) for value in selected.plan.get("test_centers", [])}
        if not column or column not in case_df.columns or not centers:
            raise ValueError("Center-holdout plan requires dataset.center_column and test_centers.")
        test = case_df.loc[case_df[column].astype(str).isin(centers)].copy()
        trainval = case_df.loc[~case_df["case_id"].astype(str).isin(test["case_id"].astype(str))].copy()
    elif selected.strategy == "train_val_test_holdout":
        labels = case_df["stratum"]
        trainval, test = train_test_split(
            case_df,
            train_size=task.train_size + task.val_size,
            random_state=task.split_seed,
            stratify=_safe_holdout_stratify(labels, task.test_size),
        )
    else:
        raise ValueError(f"Plan {selected.plan_id} has strategy {selected.strategy}; use run-cv for CV plans.")
    if trainval.empty or test.empty:
        raise ValueError(f"Plan {selected.plan_id} produced an empty train/val or test split.")
    train, val = _split_train_val(trainval, task, task.split_seed)
    return {
        **{str(value): "train" for value in train["case_id"]},
        **{str(value): "val" for value in val["case_id"]},
        **{str(value): "test" for value in test["case_id"]},
    }
