from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


CASE_RE = re.compile(r"^(?P<case>[A-Za-z0-9]+)-")


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_csv: Path
    h5_paths_csv: Path
    metadata_json: Path


@dataclass(frozen=True)
class KFoldArtifacts:
    fold_dirs: list[Path]
    metadata_json: Path


def case_id_from_h5(path: Path) -> str:
    match = CASE_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot infer case_id from H5 filename: {path.name}")
    return match.group("case")


def inspect_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "features" not in h5:
            raise ValueError(f"{path} does not contain a 'features' dataset")
        features = h5["features"]
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
    return {"shape": shape, "n_patches": n_patches, "in_dim": in_dim, "keys": keys}


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


def _load_cptac_slide_table(
    data_dir: str | Path,
    labels_csv: str | Path,
    label_column: str = "PAM50",
    min_class_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], Counter, int, dict[str, Any]]:
    data_dir = Path(data_dir)
    labels_csv = Path(labels_csv)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No H5 files found under {data_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    labels = pd.read_csv(labels_csv)
    if "case_id" not in labels.columns:
        raise ValueError("labels_csv must contain a case_id column")
    if label_column not in labels.columns:
        raise ValueError(f"labels_csv does not contain label column {label_column!r}")

    labels = labels[["case_id", label_column]].dropna().copy()
    labels["case_id"] = labels["case_id"].astype(str)
    labels[label_column] = labels[label_column].astype(str)
    counts = labels[label_column].value_counts()
    keep_labels = set(counts[counts >= min_class_count].index)
    labels = labels[labels[label_column].isin(keep_labels)].copy()

    label_names = sorted(labels[label_column].unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    labels["label"] = labels[label_column].map(label_to_id).astype(int)

    label_map = labels.set_index("case_id")[["label", label_column]].to_dict("index")
    slide_rows = []
    missing_cases = Counter()
    for path in h5_paths:
        case_id = case_id_from_h5(path)
        if case_id not in label_map:
            missing_cases[case_id] += 1
            continue
        item = label_map[case_id]
        slide_rows.append(
            {
                "case_id": case_id,
                "slide_path": str(path),
                "label": int(item["label"]),
                "label_name": item[label_column],
            }
        )

    slide_df = pd.DataFrame(slide_rows)
    if slide_df.empty:
        raise ValueError("No H5 slides matched the labels table")

    case_df = slide_df[["case_id", "label_name", "label"]].drop_duplicates("case_id")
    first_info = inspect_h5(Path(slide_df.iloc[0]["slide_path"]))
    return slide_df, case_df, label_to_id, missing_cases, len(h5_paths), first_info


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_cptac_slide_table(
        data_dir=data_dir,
        labels_csv=labels_csv,
        label_column=label_column,
        min_class_count=min_class_count,
    )
    splits = _split_cases(case_df, "label_name", seed, train_size, val_size, test_size)
    split_by_case = {case_id: split for split, cases in splits.items() for case_id in cases}
    slide_df["split"] = slide_df["case_id"].map(split_by_case)
    slide_df = slide_df.dropna(subset=["split"]).copy()

    dataset_csv = output_dir / "dataset.csv"
    wide = _wide_split_csv(slide_df)
    wide.to_csv(dataset_csv, index=False)

    h5_paths_csv = output_dir / "h5_paths.csv"
    pd.DataFrame({"h5_path": slide_df["slide_path"].astype(str)}).to_csv(h5_paths_csv, index=False)

    metadata = {
        "dataset": "CPTAC-BRCA",
        "label_column": label_column,
        "label_to_id": label_to_id,
        "num_classes": len(label_to_id),
        "num_h5_total": num_h5_total,
        "num_h5_matched": int(len(slide_df)),
        "num_cases": int(case_df.shape[0]),
        "missing_label_cases": dict(missing_cases),
        "feature": first_info,
        "split_counts_slides": slide_df.groupby(["split", "label_name"]).size().unstack(fill_value=0).to_dict(),
        "split_counts_cases": case_df.assign(split=case_df["case_id"].map(split_by_case))
        .groupby(["split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .to_dict(),
    }
    metadata_json = output_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_df, case_df, label_to_id, missing_cases, num_h5_total, first_info = _load_cptac_slide_table(
        data_dir=data_dir,
        labels_csv=labels_csv,
        label_column=label_column,
        min_class_count=min_class_count,
    )

    min_class = int(case_df["label_name"].value_counts().min())
    if n_splits > min_class:
        raise ValueError(
            f"n_splits={n_splits} is too high for the rarest class with {min_class} cases"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
            test_size=val_fraction_of_train,
            random_state=seed + fold_idx,
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
        _wide_split_csv(fold_slides).to_csv(fold_dir / "dataset.csv", index=False)
        fold_slides.to_csv(fold_dir / "slides_long.csv", index=False)
        fold_dirs.append(fold_dir)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "dataset_csv": str(fold_dir / "dataset.csv"),
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
    pd.DataFrame({"h5_path": slide_df["slide_path"].astype(str)}).to_csv(h5_paths_csv, index=False)
    metadata = {
        "dataset": "CPTAC-BRCA",
        "label_column": label_column,
        "label_to_id": label_to_id,
        "num_classes": len(label_to_id),
        "num_h5_total": num_h5_total,
        "num_h5_matched": int(len(slide_df)),
        "num_cases": int(case_df.shape[0]),
        "missing_label_cases": dict(missing_cases),
        "feature": first_info,
        "n_splits": n_splits,
        "val_fraction_of_train": val_fraction_of_train,
        "folds": fold_summaries,
    }
    metadata_json = output_dir / "metadata.json"
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    slide_df.to_csv(output_dir / "slides_long.csv", index=False)
    return KFoldArtifacts(fold_dirs=fold_dirs, metadata_json=metadata_json)
