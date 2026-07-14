from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from .state import json_ready, now_iso


PROB_RE = re.compile(r"^(?:prob|class|p)_(?P<class_id>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class FigureArtifact:
    name: str
    path: str
    description: str


@dataclass(frozen=True)
class FigureReport:
    generated_at: str
    case_predictions: str
    output_dir: str
    n_cases: int
    class_ids: list[int]
    primary_score: str | None
    artifacts: list[FigureArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_figure_report(
    case_predictions: str | Path,
    *,
    output_dir: str | Path | None = None,
    class_names: list[str] | None = None,
    positive_class: int = 1,
) -> tuple[FigureReport, Path, Path]:
    case_predictions = Path(case_predictions)
    output_dir = Path(output_dir) if output_dir else case_predictions.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(case_predictions)
    warnings: list[str] = []
    artifacts: list[FigureArtifact] = []
    prob_cols = _probability_columns(df)
    class_ids = [int(PROB_RE.match(col).group("class_id")) for col in prob_cols]

    if df.empty:
        warnings.append("case_predictions.csv is empty; no figures were generated.")
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        warnings.append("case_predictions.csv must contain y_true and y_pred for evaluation figures.")

    eval_df = df.dropna(subset=[col for col in ["y_true", "y_pred"] if col in df.columns]).copy()
    if not eval_df.empty and {"y_true", "y_pred"}.issubset(eval_df.columns):
        eval_df["y_true"] = eval_df["y_true"].astype(int)
        eval_df["y_pred"] = eval_df["y_pred"].astype(int)
        labels = _all_class_ids(eval_df, class_ids)
        names = _class_names(labels, class_names)
        per_class_csv = output_dir / "per_class_metrics.csv"
        _write_per_class_metrics(eval_df, labels, names, per_class_csv)
        artifacts.append(FigureArtifact("per_class_metrics", str(per_class_csv), "Per-class precision, recall, F1, and support."))

        error_csv = output_dir / "error_cases.csv"
        _write_error_cases(eval_df, prob_cols, error_csv)
        artifacts.append(FigureArtifact("error_cases", str(error_csv), "Misclassified cases with confidence columns when available."))

        cm_path = output_dir / "confusion_matrix.png"
        _plot_confusion_matrix(eval_df, labels, names, cm_path)
        artifacts.append(FigureArtifact("confusion_matrix", str(cm_path), "Case-level confusion matrix."))

        if prob_cols:
            roc_path = output_dir / "roc_curve.png"
            roc_score = _plot_roc(eval_df, prob_cols, labels, names, positive_class, roc_path, warnings)
            if roc_score is not None:
                artifacts.append(FigureArtifact("roc_curve", str(roc_path), "Case-level ROC curve."))
            if len(prob_cols) == 2 and positive_class in labels:
                calibration_path = output_dir / "calibration_curve.png"
                if _plot_calibration(eval_df, positive_class, calibration_path, warnings):
                    artifacts.append(FigureArtifact("calibration_curve", str(calibration_path), "Binary calibration curve."))
        else:
            warnings.append("No probability columns found, so ROC and calibration figures were skipped.")
    else:
        warnings.append("No evaluable case-level rows found.")

    report = FigureReport(
        generated_at=now_iso(),
        case_predictions=str(case_predictions),
        output_dir=str(output_dir),
        n_cases=int(len(df)),
        class_ids=class_ids,
        primary_score=_primary_score(eval_df, prob_cols, positive_class, warnings) if not eval_df.empty else None,
        artifacts=artifacts,
        warnings=warnings,
    )
    json_path = output_dir / "figure_report.json"
    md_path = output_dir / "figure_report.md"
    json_path.write_text(json.dumps(json_ready(asdict(report)), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report, json_path, md_path


def _probability_columns(df: pd.DataFrame) -> list[str]:
    cols = [column for column in df.columns if PROB_RE.match(str(column))]
    return sorted(cols, key=lambda col: int(PROB_RE.match(str(col)).group("class_id")))


def _all_class_ids(df: pd.DataFrame, class_ids: list[int]) -> list[int]:
    ids = set(class_ids)
    if "y_true" in df:
        ids.update(int(x) for x in df["y_true"].dropna().astype(int).tolist())
    if "y_pred" in df:
        ids.update(int(x) for x in df["y_pred"].dropna().astype(int).tolist())
    return sorted(ids)


def _class_names(labels: list[int], class_names: list[str] | None) -> list[str]:
    if not class_names:
        return [str(label) for label in labels]
    return [class_names[label] if 0 <= label < len(class_names) else str(label) for label in labels]


def _write_per_class_metrics(df: pd.DataFrame, labels: list[int], names: list[str], path: Path) -> None:
    report = classification_report(
        df["y_true"].astype(int),
        df["y_pred"].astype(int),
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label, name in zip(labels, names):
        row = dict(report.get(name, {}))
        row["class_id"] = label
        row["class_name"] = name
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_error_cases(df: pd.DataFrame, prob_cols: list[str], path: Path) -> None:
    errors = df.loc[df["y_true"].astype(int) != df["y_pred"].astype(int)].copy()
    if prob_cols and not errors.empty:
        probs = errors[prob_cols].astype(float)
        errors["pred_confidence"] = probs.max(axis=1)
        errors["true_class_probability"] = [
            row.get(f"prob_{int(y_true)}", np.nan) for (_, row), y_true in zip(errors.iterrows(), errors["y_true"].astype(int))
        ]
    keep = [
        col
        for col in [
            "experiment",
            "run_id",
            "model_name",
            "fold",
            "split",
            "case_id",
            "n_slides",
            "y_true",
            "y_pred",
            "pred_confidence",
            "true_class_probability",
            *prob_cols,
        ]
        if col in errors.columns
    ]
    errors[keep].to_csv(path, index=False)


def _plot_confusion_matrix(df: pd.DataFrame, labels: list[int], names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(df["y_true"].astype(int), df["y_pred"].astype(int), labels=labels)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.9), max(3.5, len(labels) * 0.8)))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Case-Level Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_roc(
    df: pd.DataFrame,
    prob_cols: list[str],
    labels: list[int],
    names: list[str],
    positive_class: int,
    path: Path,
    warnings: list[str],
) -> float | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(set(df["y_true"].astype(int))) < 2:
        warnings.append("ROC skipped because y_true contains fewer than two classes.")
        return None
    y_true = df["y_true"].astype(int).to_numpy()
    y_score = df[prob_cols].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scores: list[float] = []
    try:
        if len(prob_cols) == 2:
            pos_col = f"prob_{positive_class}"
            if pos_col not in prob_cols:
                warnings.append(f"ROC skipped because positive class column {pos_col!r} is missing.")
                plt.close(fig)
                return None
            fpr, tpr, _thresholds = roc_curve((y_true == positive_class).astype(int), df[pos_col].astype(float))
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{positive_class} AUC={score:.3f}")
            scores.append(float(score))
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for idx, (label, name) in enumerate(zip(labels, names)):
                col = f"prob_{label}"
                if col not in prob_cols:
                    continue
                if y_bin[:, idx].min() == y_bin[:, idx].max():
                    continue
                fpr, tpr, _thresholds = roc_curve(y_bin[:, idx], df[col].astype(float))
                score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name} AUC={score:.3f}")
                scores.append(float(score))
    except ValueError as exc:
        warnings.append(f"ROC skipped: {exc}")
        plt.close(fig)
        return None
    if not scores:
        warnings.append("ROC skipped because no valid one-vs-rest curves could be computed.")
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Case-Level ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return float(np.mean(scores))


def _plot_calibration(df: pd.DataFrame, positive_class: int, path: Path, warnings: list[str]) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_col = f"prob_{positive_class}"
    if pos_col not in df.columns:
        warnings.append(f"Calibration skipped because {pos_col!r} is missing.")
        return False
    y_true = (df["y_true"].astype(int) == positive_class).astype(int)
    if y_true.nunique() < 2:
        warnings.append("Calibration skipped because the positive-class target has fewer than two values.")
        return False
    frac_pos, mean_pred = calibration_curve(y_true, df[pos_col].astype(float), n_bins=min(10, max(2, len(df) // 3)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title("Case-Level Calibration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _primary_score(df: pd.DataFrame, prob_cols: list[str], positive_class: int, warnings: list[str]) -> str | None:
    if df.empty or not prob_cols:
        return None
    if len(prob_cols) == 2 and f"prob_{positive_class}" in df:
        try:
            fpr, tpr, _thresholds = roc_curve((df["y_true"].astype(int) == positive_class).astype(int), df[f"prob_{positive_class}"].astype(float))
            return f"auc={auc(fpr, tpr):.4f}"
        except ValueError as exc:
            warnings.append(f"Primary score unavailable: {exc}")
    return None


def _render_markdown(report: FigureReport) -> str:
    lines = [
        "# Figure Report",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Case predictions: `{report.case_predictions}`",
        f"- Cases: `{report.n_cases}`",
        f"- Classes: `{report.class_ids}`",
        f"- Primary score: `{report.primary_score}`",
        "",
        "## Artifacts",
        "",
        "| Name | Path | Description |",
        "|---|---|---|",
    ]
    for item in report.artifacts:
        lines.append(f"| {item.name} | `{item.path}` | {item.description} |")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in report.warnings])
    return "\n".join(lines)
