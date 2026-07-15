from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .specs import DatasetSpec, TaskSpec, specs_to_payload


CASE_RE = re.compile(r"^(?P<case>[A-Za-z0-9]+)-")
LABEL_ID_COL = "_auto_mil_label_id"


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_csv: Path
    h5_paths_csv: Path
    metadata_json: Path


@dataclass(frozen=True)
class KFoldArtifacts:
    fold_dirs: list[Path]
    metadata_json: Path


def case_id_from_h5(path: Path, case_id_regex: str = CASE_RE.pattern) -> str:
    match = re.compile(case_id_regex).match(path.name)
    if not match:
        raise ValueError(f"Cannot infer case_id from H5 filename: {path.name}")
    return match.group("case")


def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _array_from_pt_payload(payload: Any, key: str) -> np.ndarray:
    if isinstance(payload, dict):
        if key not in payload:
            raise ValueError(f"PT payload does not contain key {key!r}")
        value = payload[key]
    else:
        if key != "features":
            raise ValueError(f"PT tensor payload only supports feature_key='features', got {key!r}")
        value = payload
    return np.asarray(value)


def inspect_h5(path: Path, feature_key: str = "features", coords_key: str | None = None) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if feature_key not in h5:
            raise ValueError(f"{path} does not contain a {feature_key!r} dataset")
        features = h5[feature_key]
        shape = tuple(int(x) for x in features.shape)
        if len(shape) == 3:
            n_patches = shape[1]
            in_dim = shape[2]
        elif len(shape) == 2:
            n_patches = shape[0]
            in_dim = shape[1]
        else:
            raise ValueError(f"Unsupported features shape in {path}: {shape}")
        keys = sorted(list(h5.keys()))
        coords_shape = None
        if coords_key and coords_key in h5:
            coords_shape = tuple(int(x) for x in h5[coords_key].shape)
    return {
        "format": "h5",
        "feature_key": feature_key,
        "coords_key": coords_key,
        "shape": shape,
        "n_patches": n_patches,
        "in_dim": in_dim,
        "keys": keys,
        "coords_shape": coords_shape,
    }


def inspect_pt(path: Path, feature_key: str = "features", coords_key: str | None = None) -> dict[str, Any]:
    payload = _load_pt(path)
    features = _array_from_pt_payload(payload, feature_key)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    elif features.ndim == 3:
        features = features.reshape(-1, features.shape[-1])
    if features.ndim != 2:
        raise ValueError(f"Unsupported features shape in {path}: {features.shape}")
    coords_shape = None
    if coords_key and isinstance(payload, dict) and coords_key in payload:
        coords = np.asarray(payload[coords_key])
        coords_shape = tuple(int(x) for x in coords.shape)
    keys = sorted(payload.keys()) if isinstance(payload, dict) else ["features"]
    return {
        "format": "pt",
        "feature_key": feature_key,
        "coords_key": coords_key,
        "shape": tuple(int(x) for x in features.shape),
        "n_patches": int(features.shape[0]),
        "in_dim": int(features.shape[1]),
        "keys": keys,
        "coords_shape": coords_shape,
    }


def inspect_feature(path: Path, dataset: DatasetSpec) -> dict[str, Any]:
    fmt = dataset.feature.format.lower()
    if fmt == "h5":
        return inspect_h5(path, feature_key=dataset.feature.feature_key, coords_key=dataset.feature.coords_key)
    if fmt == "pt":
        return inspect_pt(path, feature_key=dataset.feature.feature_key, coords_key=dataset.feature.coords_key)
    raise NotImplementedError(f"Unsupported feature format: {dataset.feature.format}")


def _split_cases(
    case_df: pd.DataFrame,
    label_col: str,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> dict[str, set[str]]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    labels = case_df[label_col].astype(str)
    stratify = labels if labels.value_counts().min() >= 2 else None
    train_df, holdout_df = train_test_split(
        case_df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    holdout_labels = holdout_df[label_col].astype(str)
    holdout_stratify = holdout_labels if holdout_labels.value_counts().min() >= 2 else None
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=relative_val,
        random_state=seed,
        stratify=holdout_stratify,
    )
    return {
        "train": set(train_df["case_id"].astype(str)),
        "val": set(val_df["case_id"].astype(str)),
        "test": set(test_df["case_id"].astype(str)),
    }


def _wide_split_csv(slide_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    max_len = 0
    for split in ("train", "val", "test"):
        sub = slide_df[slide_df["split"] == split].reset_index(drop=True)
        max_len = max(max_len, len(sub))
        parts.append((split, sub))

    rows: dict[str, list[Any]] = {}
    for split, sub in parts:
        paths = sub["slide_path"].astype(str).tolist()
        labels = sub["label"].astype(int).tolist()
        rows[f"{split}_slide_path"] = paths + [""] * (max_len - len(paths))
        rows[f"{split}_label"] = labels + [""] * (max_len - len(labels))
    return pd.DataFrame(rows)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "case"


def _read_feature_array(path: Path, dataset: DatasetSpec) -> np.ndarray:
    feature_key = dataset.feature.feature_key
    if dataset.feature.format.lower() == "pt":
        features = _array_from_pt_payload(_load_pt(path), feature_key)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        elif features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if features.ndim != 2:
            raise ValueError(f"Unsupported feature shape in {path}: {features.shape}")
        return features
    with h5py.File(path, "r") as h5:
        if feature_key not in h5:
            raise ValueError(f"{path} does not contain feature dataset {feature_key!r}")
        features = np.asarray(h5[feature_key])
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    elif features.ndim == 3:
        features = features.reshape(-1, features.shape[-1])
    if features.ndim != 2:
        raise ValueError(f"Unsupported feature shape in {path}: {features.shape}")
    return features


def _read_coords_array(path: Path, dataset: DatasetSpec) -> np.ndarray | None:
    coords_key = dataset.feature.coords_key
    if not coords_key:
        return None
    if dataset.feature.format.lower() == "pt":
        payload = _load_pt(path)
        if not isinstance(payload, dict) or coords_key not in payload:
            return None
        coords = np.asarray(payload[coords_key])
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
        elif coords.ndim == 3:
            coords = coords.reshape(-1, coords.shape[-1])
        return coords if coords.ndim == 2 else None
    with h5py.File(path, "r") as h5:
        if coords_key not in h5:
            return None
        coords = np.asarray(h5[coords_key])
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    elif coords.ndim == 3:
        coords = coords.reshape(-1, coords.shape[-1])
    return coords if coords.ndim == 2 else None


def _sample_rows_deterministic(array: np.ndarray, max_rows: int | None, key: str) -> tuple[np.ndarray, np.ndarray | None]:
    if max_rows is None or max_rows <= 0 or array.shape[0] <= max_rows:
        return array, None
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(array.shape[0], size=max_rows, replace=False))
    return array[indices], indices


def _cache_sampled_slide_features(
    slide_df: pd.DataFrame,
    output_dir: Path,
    dataset: DatasetSpec,
    max_patches_per_bag: int | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if max_patches_per_bag is None or max_patches_per_bag <= 0 or dataset.bag_level != "slide":
        return slide_df, None

    cache_dir = output_dir / f"feature_cache_max{max_patches_per_bag}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"
    manifest_by_cache: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        if "cache_path" in manifest_df.columns:
            manifest_by_cache = {
                str(row["cache_path"]): row for row in manifest_df.to_dict("records")
            }
    cached_rows = []
    manifest_rows = []
    for row in slide_df.to_dict("records"):
        source_path = Path(str(row["slide_path"]))
        path_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        cache_path = cache_dir / f"{_safe_filename(str(row['case_id']))}_{path_hash}.pt"
        existing = manifest_by_cache.get(str(cache_path))
        if cache_path.exists() and existing is not None:
            updated = dict(row)
            updated["source_slide_path"] = str(source_path)
            updated["slide_path"] = str(cache_path)
            updated["n_patches_original"] = int(existing.get("n_patches_original", 0) or 0)
            updated["n_patches"] = int(existing.get("n_patches_cached", max_patches_per_bag) or max_patches_per_bag)
            cached_rows.append(updated)
            manifest_rows.append(existing)
            continue

        features = _read_feature_array(source_path, dataset)
        sampled_features, indices = _sample_rows_deterministic(
            features,
            int(max_patches_per_bag),
            str(source_path),
        )
        payload: dict[str, Any] = {"features": sampled_features}
        coords = _read_coords_array(source_path, dataset)
        coords_cached = False
        if coords is not None and coords.shape[0] == features.shape[0]:
            payload["coords"] = coords if indices is None else coords[indices]
            coords_cached = True
        if not cache_path.exists():
            torch.save(payload, cache_path)
        updated = dict(row)
        updated["source_slide_path"] = str(source_path)
        updated["slide_path"] = str(cache_path)
        updated["n_patches_original"] = int(features.shape[0])
        updated["n_patches"] = int(sampled_features.shape[0])
        cached_rows.append(updated)
        manifest_rows.append(
            {
                "case_id": row["case_id"],
                "cache_path": str(cache_path),
                "source_slide_path": str(source_path),
                "n_patches_original": int(features.shape[0]),
                "n_patches_cached": int(sampled_features.shape[0]),
                "coords_cached": bool(coords_cached),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata = {
        "feature_cache_dir": str(cache_dir),
        "feature_cache_manifest": str(manifest_path),
        "max_patches_per_bag": int(max_patches_per_bag),
        "num_cached_bags": int(len(cached_rows)),
    }
    return pd.DataFrame(cached_rows), metadata


def _case_bag_table(slide_df: pd.DataFrame, output_dir: Path, dataset: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    bag_dir = output_dir / "case_bags"
    bag_dir.mkdir(parents=True, exist_ok=True)
    bag_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for case_id, group in slide_df.sort_values("slide_path").groupby("case_id", sort=True):
        labels = sorted({int(label) for label in group["label"].tolist()})
        splits = sorted({str(split) for split in group["split"].tolist()})
        if len(labels) != 1:
            raise ValueError(f"Case {case_id} has inconsistent labels across slides: {labels}")
        if len(splits) != 1:
            raise ValueError(f"Case {case_id} appears in multiple splits: {splits}")
        feature_arrays = []
        coord_arrays = []
        coord_complete = True
        source_paths = [Path(path) for path in group["slide_path"].astype(str).tolist()]
        for source_path in source_paths:
            features = _read_feature_array(source_path, dataset)
            feature_arrays.append(features)
            coords = _read_coords_array(source_path, dataset)
            if coords is None or coords.shape[0] != features.shape[0]:
                coord_complete = False
            else:
                coord_arrays.append(coords)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "case_bag_path": str(bag_dir / f"{_safe_filename(str(case_id))}.h5"),
                    "source_slide_path": str(source_path),
                    "n_patches": int(features.shape[0]),
                }
            )
        case_features = np.concatenate(feature_arrays, axis=0)
        bag_path = bag_dir / f"{_safe_filename(str(case_id))}.h5"
        with h5py.File(bag_path, "w") as h5:
            h5.create_dataset("features", data=case_features)
            if coord_complete and coord_arrays:
                h5.create_dataset("coords", data=np.concatenate(coord_arrays, axis=0))
            h5.attrs["case_id"] = str(case_id)
            h5.attrs["source_slide_count"] = int(len(source_paths))
            h5.attrs["source_slides_json"] = json.dumps([str(path) for path in source_paths])
        first = group.iloc[0]
        bag_rows.append(
            {
                "case_id": case_id,
                "slide_path": str(bag_path),
                "label": int(first["label"]),
                "label_name": first["label_name"],
                "split": first["split"],
                "n_source_slides": int(len(source_paths)),
                "n_patches": int(case_features.shape[0]),
            }
        )
    return pd.DataFrame(bag_rows), pd.DataFrame(manifest_rows)


def _training_bag_table(slide_df: pd.DataFrame, output_dir: Path, dataset: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if dataset.bag_level == "slide":
        return slide_df.copy(), None
    if dataset.bag_level != "case":
        raise ValueError(f"Unsupported dataset.bag_level={dataset.bag_level!r}; expected 'case' or 'slide'.")
    return _case_bag_table(slide_df, output_dir, dataset)


def _read_labels_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if sheet:
            return pd.read_excel(path, sheet_name=sheet)
        with pd.ExcelFile(path) as workbook:
            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet_name)
                if not frame.empty:
                    return frame
        raise ValueError(f"No non-empty sheet found in {path}")
    return pd.read_csv(path)


def _default_feature_glob(dataset: DatasetSpec) -> str:
    fmt = dataset.feature.format.lower()
    if fmt == "pt":
        return "**/features/*.pt"
    if fmt == "h5":
        return "*.h5"
    return f"**/*.{fmt}"


def _discover_feature_paths(dataset: DatasetSpec) -> list[Path]:
    pattern = dataset.feature.feature_glob or _default_feature_glob(dataset)
    return sorted(path for path in dataset.data_dir.glob(pattern) if path.is_file())


def _normalize_feature_path(path_value: Any, dataset: DatasetSpec) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return dataset.data_dir / path


def _feature_paths_from_labels(labels: pd.DataFrame, dataset: DatasetSpec) -> list[Path]:
    column = dataset.slide_path_column
    if not column or column not in labels.columns:
        return []
    paths = [_normalize_feature_path(value, dataset) for value in labels[column].dropna().tolist()]
    return sorted(dict.fromkeys(paths))


def _explicit_case_by_path(labels: pd.DataFrame, dataset: DatasetSpec) -> dict[str, str]:
    column = dataset.slide_path_column
    if not column or column not in labels.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in labels.dropna(subset=[column, "case_id"]).iterrows():
        mapping[str(_normalize_feature_path(row[column], dataset))] = str(row["case_id"])
    return mapping


def _apply_classification_label_transform(labels: pd.DataFrame, task: TaskSpec) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    label_column = task.outcome_column
    if task.label_threshold is None:
        labels[label_column] = labels[label_column].astype(str)
        return labels, label_column, {}
    transformed_column = f"{label_column}__threshold"
    values = pd.to_numeric(labels[label_column], errors="coerce")
    direction = task.label_threshold_direction.lower()
    if direction in {"ge", ">=", "gte"}:
        positive = values >= float(task.label_threshold)
        rule = f"{label_column} >= {task.label_threshold}"
    elif direction in {"gt", ">"}:
        positive = values > float(task.label_threshold)
        rule = f"{label_column} > {task.label_threshold}"
    elif direction in {"le", "<=", "lte"}:
        positive = values <= float(task.label_threshold)
        rule = f"{label_column} <= {task.label_threshold}"
    elif direction in {"lt", "<"}:
        positive = values < float(task.label_threshold)
        rule = f"{label_column} < {task.label_threshold}"
    else:
        raise ValueError(f"Unsupported label_threshold_direction={task.label_threshold_direction!r}")
    labels = labels.copy()
    labels[transformed_column] = np.where(positive, task.positive_label, task.negative_label)
    labels.loc[values.isna(), transformed_column] = np.nan
    return labels, transformed_column, {
        "source_label_column": label_column,
        "transformed_label_column": transformed_column,
        "label_threshold": float(task.label_threshold),
        "label_threshold_direction": task.label_threshold_direction,
        "negative_label": task.negative_label,
        "positive_label": task.positive_label,
        "rule": rule,
    }


def _load_h5_classification_slide_table(
    dataset: DatasetSpec,
    task: TaskSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], Counter, int, dict[str, Any]]:
    task.validate_for_mil_baseline()
    if dataset.feature.format.lower() not in {"h5", "pt"}:
        raise NotImplementedError(f"Unsupported feature format for MIL_BASELINE preparation: {dataset.feature.format}")
    data_dir = dataset.data_dir
    labels_csv = dataset.labels_csv
    label_column = task.outcome_column

    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    labels = _read_labels_table(labels_csv, dataset.labels_sheet)
    if dataset.case_id_column not in labels.columns:
        raise ValueError(f"labels_csv must contain a {dataset.case_id_column!r} column")
    if label_column not in labels.columns:
        raise ValueError(f"labels_csv does not contain label column {label_column!r}")
    labels, effective_label_column, label_transform = _apply_classification_label_transform(labels, task)

    keep_columns = [dataset.case_id_column, label_column, effective_label_column]
    optional_columns = [
        dataset.center_column,
        dataset.cohort_column,
        dataset.external_test_column,
        dataset.slide_path_column,
    ]
    keep_columns.extend([col for col in optional_columns if col and col in labels.columns])
    keep_columns = list(dict.fromkeys(keep_columns))
    labels = labels[keep_columns].dropna(subset=[dataset.case_id_column, label_column]).copy()
    labels = labels.dropna(subset=[effective_label_column]).copy()
    labels = labels.rename(columns={dataset.case_id_column: "case_id"})
    labels["case_id"] = labels["case_id"].astype(str)
    labels[effective_label_column] = labels[effective_label_column].astype(str)
    counts = labels[effective_label_column].value_counts()
    keep_labels = set(counts[counts >= task.min_class_count].index)
    labels = labels[labels[effective_label_column].isin(keep_labels)].copy()

    if label_transform:
        label_names = [task.negative_label, task.positive_label]
        label_names = [name for name in label_names if name in set(labels[effective_label_column])]
    else:
        label_names = sorted(labels[effective_label_column].unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    labels[LABEL_ID_COL] = labels[effective_label_column].map(label_to_id).astype(int)

    label_map_columns = [LABEL_ID_COL, effective_label_column]
    for optional in (dataset.center_column, dataset.cohort_column, dataset.external_test_column):
        if optional and optional in labels.columns and optional not in label_map_columns:
            label_map_columns.append(optional)
    label_map = labels.set_index("case_id")[label_map_columns].to_dict("index")
    slide_rows = []
    missing_cases = Counter()
    missing_feature_paths = 0
    feature_paths = _feature_paths_from_labels(labels, dataset) if dataset.slide_path_column else _discover_feature_paths(dataset)
    if not feature_paths:
        raise FileNotFoundError(f"No {dataset.feature.format.upper()} feature files found for {dataset.name} under {data_dir}")
    explicit_by_path = _explicit_case_by_path(labels, dataset) if dataset.slide_path_column else {}
    for path in feature_paths:
        case_id = explicit_by_path.get(str(path)) or case_id_from_h5(path, dataset.feature.case_id_regex)
        if case_id not in label_map:
            missing_cases[case_id] += 1
            continue
        if not path.exists():
            missing_feature_paths += 1
            continue
        item = label_map[case_id]
        row = {
            "case_id": case_id,
            "slide_path": str(path),
            "label": int(item[LABEL_ID_COL]),
            "label_name": item[effective_label_column],
        }
        for optional in ("center_column", "cohort_column", "external_test_column"):
            source_col = getattr(dataset, optional)
            if source_col and source_col in item:
                row[source_col] = item[source_col]
        slide_rows.append(row)

    slide_df = pd.DataFrame(slide_rows)
    if slide_df.empty:
        raise ValueError("No H5 slides matched the labels table")

    case_columns = ["case_id", "label_name", "label"]
    for optional in (dataset.center_column, dataset.cohort_column, dataset.external_test_column):
        if optional and optional in slide_df.columns:
            case_columns.append(optional)
    case_df = slide_df[case_columns].drop_duplicates("case_id")
    first_info = inspect_feature(Path(slide_df.iloc[0]["slide_path"]), dataset)
    first_info["missing_feature_paths"] = int(missing_feature_paths)
    if label_transform:
        first_info["label_transform"] = label_transform
    return slide_df, case_df, label_to_id, missing_cases, len(feature_paths), first_info


def _load_cptac_slide_table(
    data_dir: str | Path,
    labels_csv: str | Path,
    label_column: str = "PAM50",
    min_class_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], Counter, int, dict[str, Any]]:
    dataset = DatasetSpec(name="CPTAC-BRCA", data_dir=Path(data_dir), labels_csv=Path(labels_csv))
    task = TaskSpec(kind="classification", label_column=label_column, min_class_count=min_class_count)
    return _load_h5_classification_slide_table(dataset, task)


def _metadata_base(
    dataset: DatasetSpec,
    task: TaskSpec,
    label_to_id: dict[str, int],
    missing_cases: Counter,
    num_feature_total: int,
    slide_df: pd.DataFrame,
    case_df: pd.DataFrame,
    first_info: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "dataset": dataset.name,
        "label_column": task.label_column,
        "task": specs_to_payload(task, dataset)["task"],
        "dataset_spec": specs_to_payload(task, dataset)["dataset"],
        "label_to_id": label_to_id,
        "num_classes": len(label_to_id),
        "num_feature_total": num_feature_total,
        "num_feature_matched": int(len(slide_df)),
        "num_h5_total": num_feature_total,
        "num_h5_matched": int(len(slide_df)),
        "num_cases": int(case_df.shape[0]),
        "missing_label_cases": dict(missing_cases),
        "feature": first_info,
    }
    for column in (dataset.center_column, dataset.cohort_column, dataset.external_test_column):
        if column and column in case_df.columns:
            metadata[f"{column}_counts"] = case_df[column].astype(str).value_counts().to_dict()
    return metadata


def prepare_cptac_brca(
    data_dir: str | Path,
    labels_csv: str | Path,
    output_dir: str | Path,
    label_column: str = "PAM50",
    min_class_count: int = 2,
    seed: int = 2024,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> DatasetArtifacts:
    dataset = DatasetSpec(name="CPTAC-BRCA", data_dir=Path(data_dir), labels_csv=Path(labels_csv))
    task = TaskSpec(
        kind="classification",
        label_column=label_column,
        min_class_count=min_class_count,
        split_seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    return prepare_dataset(dataset=dataset, task=task, output_dir=output_dir)


def prepare_dataset(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
) -> DatasetArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_h5_classification_slide_table(
        dataset,
        task,
    )
    splits = _split_cases(case_df, "label_name", task.split_seed, task.train_size, task.val_size, task.test_size)
    split_by_case = {case_id: split for split, cases in splits.items() for case_id in cases}
    return prepare_dataset_from_case_splits(
        dataset=dataset,
        task=task,
        output_dir=output_dir,
        split_by_case=split_by_case,
        loaded=(slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info),
    )


def prepare_dataset_from_case_splits(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    split_by_case: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
    loaded: tuple[pd.DataFrame, pd.DataFrame, dict[str, int], Counter, int, dict[str, Any]] | None = None,
) -> DatasetArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if loaded is None:
        slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_h5_classification_slide_table(
            dataset,
            task,
        )
    else:
        slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = loaded
    slide_df["split"] = slide_df["case_id"].map(split_by_case)
    slide_df = slide_df.dropna(subset=["split"]).copy()
    bag_df, bag_manifest = _training_bag_table(slide_df, output_dir, dataset)

    dataset_csv = output_dir / "dataset.csv"
    wide = _wide_split_csv(bag_df)
    wide.to_csv(dataset_csv, index=False)

    h5_paths_csv = output_dir / "h5_paths.csv"
    pd.DataFrame({"h5_path": bag_df["slide_path"].astype(str)}).to_csv(h5_paths_csv, index=False)

    metadata = _metadata_base(dataset, task, label_to_id, missing_cases, num_h5_total, slide_df, case_df, first_info)
    metadata.update(
        {
            "bag_level": dataset.bag_level,
            "num_training_bags": int(len(bag_df)),
            "bag_counts": bag_df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
            "split_counts_slides": slide_df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
            "split_counts_cases": case_df.assign(split=case_df["case_id"].map(split_by_case))
            .groupby(["split", "label_name"])
            .size()
            .unstack(fill_value=0)
            .to_dict(),
        }
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_json = output_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
    bag_df.to_csv(output_dir / "bags_long.csv", index=False)
    if bag_manifest is not None:
        bag_manifest.to_csv(output_dir / "case_bag_manifest.csv", index=False)
    return DatasetArtifacts(dataset_csv=dataset_csv, h5_paths_csv=h5_paths_csv, metadata_json=metadata_json)


def prepare_cptac_brca_kfold(
    data_dir: str | Path,
    labels_csv: str | Path,
    output_dir: str | Path,
    label_column: str = "PAM50",
    min_class_count: int = 2,
    seed: int = 2024,
    n_splits: int = 5,
    val_fraction_of_train: float = 0.2,
) -> KFoldArtifacts:
    dataset = DatasetSpec(name="CPTAC-BRCA", data_dir=Path(data_dir), labels_csv=Path(labels_csv))
    task = TaskSpec(
        kind="classification",
        label_column=label_column,
        min_class_count=min_class_count,
        split_seed=seed,
        cv_val_fraction_of_train=val_fraction_of_train,
    )
    return prepare_dataset_kfold(dataset=dataset, task=task, output_dir=output_dir, n_splits=n_splits)


def prepare_dataset_kfold(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    n_splits: int = 5,
    max_patches_per_bag: int | None = None,
) -> KFoldArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_h5_classification_slide_table(
        dataset,
        task,
    )
    slide_df, feature_cache_metadata = _cache_sampled_slide_features(
        slide_df,
        output_dir,
        dataset,
        max_patches_per_bag,
    )

    min_class = int(case_df["label_name"].value_counts().min())
    if n_splits > min_class:
        raise ValueError(
            f"n_splits={n_splits} is too high for the rarest class with {min_class} cases"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=task.split_seed)
    fold_dirs: list[Path] = []
    fold_summaries: list[dict[str, Any]] = []
    case_df = case_df.reset_index(drop=True)
    for fold_idx, (trainval_idx, test_idx) in enumerate(
        splitter.split(case_df["case_id"], case_df["label_name"])
    ):
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
        split_by_case = {
            **{case_id: "train" for case_id in train_cases["case_id"].astype(str)},
            **{case_id: "val" for case_id in val_cases["case_id"].astype(str)},
            **{case_id: "test" for case_id in test_cases["case_id"].astype(str)},
        }
        fold_slides = slide_df.copy()
        fold_slides["split"] = fold_slides["case_id"].map(split_by_case)
        fold_slides = fold_slides.dropna(subset=["split"]).copy()

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_bags, fold_manifest = _training_bag_table(fold_slides, fold_dir, dataset)
        _wide_split_csv(fold_bags).to_csv(fold_dir / "dataset.csv", index=False)
        fold_slides.to_csv(fold_dir / "slides_long.csv", index=False)
        fold_bags.to_csv(fold_dir / "bags_long.csv", index=False)
        if fold_manifest is not None:
            fold_manifest.to_csv(fold_dir / "case_bag_manifest.csv", index=False)
        fold_dirs.append(fold_dir)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "dataset_csv": str(fold_dir / "dataset.csv"),
                "bag_level": dataset.bag_level,
                "bag_counts": (
                    fold_bags.groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
                "case_counts": (
                    case_df.assign(split=case_df["case_id"].map(split_by_case))
                    .dropna(subset=["split"])
                    .groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
                "slide_counts": (
                    fold_slides.groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
            }
    )

    h5_paths_csv = output_dir / "h5_paths.csv"
    bag_paths = []
    for fold_dir in fold_dirs:
        bags_csv = fold_dir / "bags_long.csv"
        if bags_csv.exists():
            bag_paths.extend(pd.read_csv(bags_csv)["slide_path"].astype(str).tolist())
    pd.DataFrame({"h5_path": sorted(set(bag_paths))}).to_csv(h5_paths_csv, index=False)
    metadata = _metadata_base(dataset, task, label_to_id, missing_cases, num_h5_total, slide_df, case_df, first_info)
    metadata.update(
        {
            "bag_level": dataset.bag_level,
            "n_splits": n_splits,
            "val_fraction_of_train": task.cv_val_fraction_of_train,
            "folds": fold_summaries,
        }
    )
    if feature_cache_metadata:
        metadata["feature_cache"] = feature_cache_metadata
    metadata_json = output_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
    return KFoldArtifacts(fold_dirs=fold_dirs, metadata_json=metadata_json)


def prepare_dataset_kfold_from_case_splits(
    *,
    dataset: DatasetSpec,
    task: TaskSpec,
    output_dir: str | Path,
    fold_split_by_case: list[dict[str, str]],
    extra_metadata: dict[str, Any] | None = None,
) -> KFoldArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_h5_classification_slide_table(
        dataset,
        task,
    )
    fold_dirs: list[Path] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, split_by_case in enumerate(fold_split_by_case):
        fold_slides = slide_df.copy()
        fold_slides["split"] = fold_slides["case_id"].map(split_by_case)
        fold_slides = fold_slides.dropna(subset=["split"]).copy()
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_bags, fold_manifest = _training_bag_table(fold_slides, fold_dir, dataset)
        _wide_split_csv(fold_bags).to_csv(fold_dir / "dataset.csv", index=False)
        fold_slides.to_csv(fold_dir / "slides_long.csv", index=False)
        fold_bags.to_csv(fold_dir / "bags_long.csv", index=False)
        if fold_manifest is not None:
            fold_manifest.to_csv(fold_dir / "case_bag_manifest.csv", index=False)
        fold_dirs.append(fold_dir)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "dataset_csv": str(fold_dir / "dataset.csv"),
                "bag_level": dataset.bag_level,
                "bag_counts": (
                    fold_bags.groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
                "case_counts": (
                    case_df.assign(split=case_df["case_id"].map(split_by_case))
                    .dropna(subset=["split"])
                    .groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
                "slide_counts": (
                    fold_slides.groupby(["split", "label_name"])
                    .size()
                    .unstack(fill_value=0)
                    .to_dict()
                ),
            }
        )

    h5_paths_csv = output_dir / "h5_paths.csv"
    bag_paths = []
    for fold_dir in fold_dirs:
        bags_csv = fold_dir / "bags_long.csv"
        if bags_csv.exists():
            bag_paths.extend(pd.read_csv(bags_csv)["slide_path"].astype(str).tolist())
    pd.DataFrame({"h5_path": sorted(set(bag_paths))}).to_csv(h5_paths_csv, index=False)
    metadata = _metadata_base(dataset, task, label_to_id, missing_cases, num_h5_total, slide_df, case_df, first_info)
    metadata.update(
        {
            "bag_level": dataset.bag_level,
            "n_splits": len(fold_split_by_case),
            "val_fraction_of_train": task.cv_val_fraction_of_train,
            "folds": fold_summaries,
        }
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_json = output_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
    return KFoldArtifacts(fold_dirs=fold_dirs, metadata_json=metadata_json)
