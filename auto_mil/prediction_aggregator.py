from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .data import CASE_RE
from .state import json_ready, now_iso


PROB_RE = re.compile(r"^(?:prob|class|p)_(?P<class_id>\d+)$", re.IGNORECASE)
PREDICTION_GLOBS = ["predictions.csv", "*_predictions.csv", "test_predictions.csv", "*prediction*.csv"]


@dataclass(frozen=True)
class PredictionBundle:
    generated_at: str
    root: str
    prediction_files: list[str]
    aggregation: str
    num_slide_predictions: int
    num_case_predictions: int
    class_columns: list[str]
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def discover_prediction_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    paths: set[Path] = set()
    for pattern in PREDICTION_GLOBS:
        for path in root.rglob(pattern):
            if path.is_file() and "prediction_aggregation" not in path.parts:
                paths.add(path)
    return sorted(paths)


def aggregate_predictions(
    root: str | Path,
    *,
    prediction_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    case_id_regex: str = CASE_RE.pattern,
    aggregation: str = "mean",
) -> tuple[PredictionBundle, Path, Path, Path, Path]:
    root = Path(root)
    paths = [Path(path) for path in prediction_paths] if prediction_paths else discover_prediction_files(root)
    output_dir = Path(output_dir) if output_dir else root / "prediction_aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if not paths:
        warnings.append(f"No prediction CSV files found under {root}.")
    slide_df = _load_slide_predictions(paths, root=root, case_id_regex=case_id_regex, warnings=warnings)
    case_df = _aggregate_case_predictions(slide_df, aggregation=aggregation, warnings=warnings)
    metrics = _compute_metrics(case_df, warnings=warnings)

    slide_csv = output_dir / "slide_predictions.csv"
    case_csv = output_dir / "case_predictions.csv"
    metrics_json = output_dir / "case_metrics.json"
    report_md = output_dir / "prediction_report.md"
    slide_df.to_csv(slide_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    bundle = PredictionBundle(
        generated_at=now_iso(),
        root=str(root),
        prediction_files=[str(path) for path in paths],
        aggregation=aggregation,
        num_slide_predictions=int(len(slide_df)),
        num_case_predictions=int(len(case_df)),
        class_columns=_probability_columns(slide_df),
        metrics=metrics,
        warnings=warnings,
    )
    metrics_json.write_text(json.dumps(json_ready(asdict(bundle)), indent=2, ensure_ascii=True), encoding="utf-8")
    report_md.write_text(_render_markdown(bundle, case_df), encoding="utf-8")
    return bundle, slide_csv, case_csv, metrics_json, report_md


def _load_slide_predictions(paths: list[Path], *, root: Path, case_id_regex: str, warnings: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            warnings.append(f"Could not read prediction file {path}: {exc}")
            continue
        if frame.empty:
            warnings.append(f"Prediction file is empty: {path}")
            continue
        frame = frame.copy()
        frame["prediction_file"] = str(path)
        frame["experiment"] = _experiment_name(root, path)
        frames.append(_normalize_prediction_frame(frame, path=path, case_id_regex=case_id_regex, warnings=warnings))
    if not frames:
        return pd.DataFrame(columns=["case_id", "slide_id", "slide_path", "y_true", "y_pred", "prediction_file", "experiment"])
    out = pd.concat(frames, ignore_index=True)
    prob_cols = _probability_columns(out)
    keep = [
        "experiment",
        "run_id",
        "model_name",
        "fold",
        "split",
        "case_id",
        "slide_id",
        "slide_path",
        "y_true",
        "y_pred",
        *prob_cols,
        "prediction_file",
    ]
    for column in keep:
        if column not in out.columns:
            out[column] = None
    return out[keep]


def _normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    case_id_regex: str,
    warnings: list[str],
) -> pd.DataFrame:
    rename_map = {}
    aliases = {
        "case_id": ["case_id", "patient_id", "subject_id"],
        "slide_path": ["slide_path", "wsi_path", "h5_path", "path", "file"],
        "slide_id": ["slide_id", "wsi_id", "slide", "filename"],
        "y_true": ["y_true", "label", "true_label", "target"],
        "y_pred": ["y_pred", "pred", "pred_label", "prediction"],
        "run_id": ["run_id", "recipe_id"],
        "model_name": ["model_name", "model", "variant"],
        "fold": ["fold", "fold_idx"],
    }
    lower_to_original = {column.lower(): column for column in frame.columns}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                rename_map[original] = canonical
                break
    frame = frame.rename(columns=rename_map)
    _normalize_probability_columns(frame)

    if "slide_path" not in frame.columns and "slide_id" in frame.columns:
        frame["slide_path"] = frame["slide_id"]
    if "slide_id" not in frame.columns and "slide_path" in frame.columns:
        frame["slide_id"] = frame["slide_path"].map(lambda value: Path(str(value)).stem)
    if "case_id" not in frame.columns:
        source = frame["slide_path"] if "slide_path" in frame.columns else frame.get("slide_id")
        if source is None:
            warnings.append(f"{path} has no case_id, slide_path, or slide_id column.")
            frame["case_id"] = None
        else:
            regex = re.compile(case_id_regex)
            frame["case_id"] = source.map(lambda value: _infer_case_id(value, regex))
            if frame["case_id"].isna().any():
                warnings.append(f"Some case_id values could not be inferred in {path}.")
    if "y_pred" not in frame.columns:
        prob_cols = _probability_columns(frame)
        if prob_cols:
            frame["y_pred"] = frame[prob_cols].astype(float).idxmax(axis=1).map(lambda col: int(PROB_RE.match(col).group("class_id")))
    return frame


def _normalize_probability_columns(frame: pd.DataFrame) -> None:
    for column in list(frame.columns):
        lower = column.lower()
        if re.match(r"^prob\d+$", lower):
            frame.rename(columns={column: f"prob_{lower[4:]}"}, inplace=True)
        elif re.match(r"^p\d+$", lower):
            frame.rename(columns={column: f"prob_{lower[1:]}"}, inplace=True)
        elif lower.startswith("class_") and lower[6:].isdigit():
            frame.rename(columns={column: f"prob_{lower[6:]}"}, inplace=True)


def _infer_case_id(value: Any, regex: re.Pattern[str]) -> str | None:
    if value is None or pd.isna(value):
        return None
    name = Path(str(value)).name
    match = regex.match(name)
    if match:
        return match.groupdict().get("case") or match.group(1)
    return Path(name).stem


def _aggregate_case_predictions(slide_df: pd.DataFrame, *, aggregation: str, warnings: list[str]) -> pd.DataFrame:
    prob_cols = _probability_columns(slide_df)
    if slide_df.empty:
        columns = ["experiment", "run_id", "model_name", "fold", "split", "case_id", "n_slides", "y_true", "y_pred", *prob_cols]
        return pd.DataFrame(columns=columns)
    if not prob_cols:
        warnings.append("No probability columns found; case-level probabilities and AUC cannot be computed.")
    group_cols = [column for column in ["experiment", "run_id", "model_name", "fold", "split", "case_id"] if column in slide_df.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in slide_df.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        row["n_slides"] = int(len(group))
        labels = [int(float(value)) for value in group["y_true"].dropna().tolist()] if "y_true" in group else []
        row["y_true"] = _single_or_majority(labels, warnings, row.get("case_id"))
        if prob_cols:
            probs = group[prob_cols].astype(float)
            if aggregation == "max":
                values = probs.max(axis=0)
            elif aggregation == "median":
                values = probs.median(axis=0)
            else:
                values = probs.mean(axis=0)
            for col in prob_cols:
                row[col] = float(values[col])
            row["y_pred"] = int(max(prob_cols, key=lambda col: row[col]).split("_")[-1])
        elif "y_pred" in group:
            preds = [int(float(value)) for value in group["y_pred"].dropna().tolist()]
            row["y_pred"] = _single_or_majority(preds, warnings, row.get("case_id"))
        rows.append(row)
    return pd.DataFrame(rows)


def _single_or_majority(values: list[int], warnings: list[str], case_id: Any) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) > 1:
        warnings.append(f"Non-identical slide labels/predictions for case {case_id}; using majority vote.")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _compute_metrics(case_df: pd.DataFrame, *, warnings: list[str]) -> dict[str, Any]:
    if case_df.empty or "y_true" not in case_df or "y_pred" not in case_df:
        return {}
    eval_df = case_df.dropna(subset=["y_true", "y_pred"]).copy()
    if eval_df.empty:
        warnings.append("No case-level rows with both y_true and y_pred.")
        return {}
    y_true = eval_df["y_true"].astype(int).to_numpy()
    y_pred = eval_df["y_pred"].astype(int).to_numpy()
    metrics: dict[str, Any] = {
        "n_cases": int(len(eval_df)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_mat": confusion_matrix(y_true, y_pred).tolist(),
    }
    prob_cols = _probability_columns(eval_df)
    if prob_cols:
        y_score = eval_df[prob_cols].astype(float).to_numpy()
        try:
            if len(prob_cols) > 2:
                labels = [int(col.split("_")[-1]) for col in prob_cols]
                metrics["macro_auc"] = float(roc_auc_score(y_true, y_score, average="macro", multi_class="ovr", labels=labels))
            else:
                metrics["macro_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
        except ValueError as exc:
            warnings.append(f"Could not compute case-level AUC: {exc}")
    return metrics


def _probability_columns(df: pd.DataFrame) -> list[str]:
    cols = [column for column in df.columns if PROB_RE.match(str(column))]
    return sorted(cols, key=lambda col: int(PROB_RE.match(str(col)).group("class_id")))


def _experiment_name(root: Path, path: Path) -> str:
    try:
        relative = path.parent.relative_to(root)
    except ValueError:
        relative = path.parent
    text = str(relative).replace("\\", "/")
    return text if text not in {"", "."} else path.parent.name


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(bundle: PredictionBundle, case_df: pd.DataFrame) -> str:
    lines = [
        "# Prediction Aggregation Report",
        "",
        f"- Generated: `{bundle.generated_at}`",
        f"- Root: `{bundle.root}`",
        f"- Prediction files: `{len(bundle.prediction_files)}`",
        f"- Slide predictions: `{bundle.num_slide_predictions}`",
        f"- Case predictions: `{bundle.num_case_predictions}`",
        f"- Aggregation: `{bundle.aggregation}`",
        "",
        "## Case-Level Metrics",
        "",
    ]
    if bundle.metrics:
        for key, value in bundle.metrics.items():
            lines.append(f"- {key}: `{_fmt(value)}`")
    else:
        lines.append("- No case-level metrics available.")
    lines.extend(["", "## Case Predictions", ""])
    preview_cols = [column for column in ["experiment", "run_id", "model_name", "fold", "split", "case_id", "n_slides", "y_true", "y_pred", *bundle.class_columns] if column in case_df.columns]
    if preview_cols and not case_df.empty:
        lines.append("| " + " | ".join(preview_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(preview_cols)) + "|")
        for _, row in case_df.head(50).iterrows():
            lines.append("| " + " | ".join(_fmt(row.get(col)) for col in preview_cols) + " |")
    else:
        lines.append("No case predictions.")
    if bundle.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in bundle.warnings])
    return "\n".join(lines)
