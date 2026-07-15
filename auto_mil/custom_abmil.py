from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .baseline_registry import assert_mil_baseline_root


class WideWSIDataset(Dataset):
    def __init__(self, dataset_csv: str | Path, split: str, use_coords: bool = False):
        self.dataset_csv = Path(dataset_csv)
        self.use_coords = use_coords
        df = pd.read_csv(self.dataset_csv)
        self.slide_paths = df[f"{split}_slide_path"].dropna().astype(str).tolist()
        self.labels = [int(float(x)) for x in df[f"{split}_label"].dropna().tolist()]

    def __len__(self) -> int:
        return len(self.slide_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.slide_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if path.endswith(".h5"):
            with h5py.File(path, "r") as h5:
                feat = torch.from_numpy(np.array(h5["features"]))
                coords = torch.from_numpy(np.array(h5["coords"])) if self.use_coords and "coords" in h5 else None
        else:
            try:
                loaded = torch.load(path)
            except Exception as exc:
                if "Weights only load failed" not in str(exc):
                    raise
                loaded = torch.load(path, weights_only=False)
            coords = None
            if isinstance(loaded, dict):
                feat = loaded.get("feats", loaded.get("features"))
                if feat is None:
                    raise ValueError(f"Unknown feature dict keys in {path}: {list(loaded)}")
                if self.use_coords and "coords" in loaded:
                    coords = loaded["coords"]
            else:
                feat = loaded
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        if self.use_coords and coords is not None:
            if len(coords.shape) == 3:
                coords = coords.squeeze(0)
            if coords.shape[0] == feat.shape[0] and coords.shape[-1] >= 2:
                coords = coords[:, :2].float()
                coords = coords - coords.min(dim=0, keepdim=True).values
                scale = coords.max(dim=0, keepdim=True).values.clamp_min(1.0)
                coords = coords / scale
                feat = torch.cat([feat.float(), coords], dim=-1)
        return feat.float(), label

    def balanced_sampler(self) -> WeightedRandomSampler:
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[label] for label in self.labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        labels: list[int],
        num_classes: int,
        gamma: float = 2.0,
        beta: float = 0.999,
    ):
        super().__init__()
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for label in labels:
            counts[label] += 1
        effective_num = 1.0 - torch.pow(torch.tensor(beta), counts.clamp_min(1.0))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-8)
        alpha = self.weights[target]
        return (alpha * (1.0 - pt).pow(self.gamma) * ce).mean()


class PrototypeHead(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, temperature: float = 0.2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.02)
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return features @ prototypes.t() / self.temperature


@dataclass
class TrainConfig:
    dataset_csv: Path
    mil_baseline_dir: Path
    output_dir: Path
    variant: str
    num_classes: int
    in_dim: int
    epochs: int
    device: int
    seed: int
    lr: float
    dropout: float
    balanced_sampler: bool
    focal_gamma: float
    focal_beta: float
    prototype_lambda: float
    prototype_temperature: float
    use_coords: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_abmil(cfg: TrainConfig) -> nn.Module:
    sys.path.insert(0, str(cfg.mil_baseline_dir))
    from modules.AB_MIL.ab_mil import AB_MIL

    return AB_MIL(
        L=512,
        D=128,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        act=nn.ReLU(),
        in_dim=cfg.in_dim,
    )


def cal_scores(probs: list[np.ndarray], labels: list[int], num_classes: int) -> dict[str, Any]:
    y_score = np.vstack(probs)
    y_true = np.array(labels)
    y_pred = y_score.argmax(axis=1)
    out: dict[str, Any] = {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_pre": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_pre": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_pre": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "linear_kappa": cohen_kappa_score(y_true, y_pred, weights="linear"),
        "confusion_mat": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        if num_classes > 2:
            out["macro_auc"] = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
            out["micro_auc"] = roc_auc_score(y_true, y_score, average="micro", multi_class="ovr")
            out["weighted_auc"] = roc_auc_score(y_true, y_score, average="weighted", multi_class="ovr")
        else:
            out["macro_auc"] = roc_auc_score(y_true, y_score[:, 1])
            out["micro_auc"] = out["macro_auc"]
            out["weighted_auc"] = out["macro_auc"]
    except ValueError:
        out["macro_auc"] = math.nan
        out["micro_auc"] = math.nan
        out["weighted_auc"] = math.nan
    return out


def evaluate(
    model: nn.Module,
    prototype_head: PrototypeHead | None,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    if prototype_head is not None:
        prototype_head.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for bag, label in loader:
            bag = bag.to(device)
            label = label.to(device)
            output = model(bag, return_WSI_feature=prototype_head is not None)
            logits = output["logits"]
            loss = criterion(logits, label)
            if prototype_head is not None:
                proto_logits = prototype_head(output["WSI_feature"])
                loss = loss + cfg.prototype_lambda * F.cross_entropy(proto_logits, label)
            losses.append(float(loss.item()))
            probs.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            labels.append(int(label.item()))
    return float(np.mean(losses)), cal_scores(probs, labels, cfg.num_classes)


def write_predictions(
    model: nn.Module,
    dataset: WideWSIDataset,
    *,
    output_path: Path,
    split: str,
    cfg: TrainConfig,
    device: torch.device,
) -> Path:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            bag, label = dataset[idx]
            logits = model(bag.unsqueeze(0).to(device))["logits"]
            prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            row: dict[str, Any] = {
                "run_id": f"{cfg.variant}_{split}",
                "model_name": cfg.variant,
                "split": split,
                "slide_path": dataset.slide_paths[idx],
                "slide_id": Path(dataset.slide_paths[idx]).stem,
                "y_true": int(label.item()),
                "y_pred": int(prob.argmax()),
            }
            for class_idx, value in enumerate(prob.tolist()):
                row[f"prob_{class_idx}"] = float(value)
            rows.append(row)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "model_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def train(cfg: TrainConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")

    train_set = WideWSIDataset(cfg.dataset_csv, "train", use_coords=cfg.use_coords)
    val_set = WideWSIDataset(cfg.dataset_csv, "val", use_coords=cfg.use_coords)
    test_set = WideWSIDataset(cfg.dataset_csv, "test", use_coords=cfg.use_coords)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    if cfg.balanced_sampler:
        train_loader = DataLoader(train_set, batch_size=1, sampler=train_set.balanced_sampler(), generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = get_abmil(cfg).to(device)
    use_focal = "FOCAL" in cfg.variant.upper()
    use_proto = "PROTO" in cfg.variant.upper()
    criterion: nn.Module
    if use_focal:
        criterion = ClassBalancedFocalLoss(
            train_set.labels,
            cfg.num_classes,
            gamma=cfg.focal_gamma,
            beta=cfg.focal_beta,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    prototype_head = PrototypeHead(cfg.num_classes, 512, cfg.prototype_temperature).to(device) if use_proto else None

    params = list(model.parameters())
    if prototype_head is not None:
        params += list(prototype_head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=1e-5)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if prototype_head is not None:
            prototype_head.train()
        train_losses = []
        t0 = time.time()
        for bag, label in train_loader:
            bag = bag.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(bag, return_WSI_feature=prototype_head is not None)
            logits = output["logits"]
            loss = criterion(logits, label)
            if prototype_head is not None:
                proto_logits = prototype_head(output["WSI_feature"])
                loss = loss + cfg.prototype_lambda * F.cross_entropy(proto_logits, label)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_metrics = evaluate(model, prototype_head, val_loader, criterion, cfg, device)
        test_loss, test_metrics = evaluate(model, prototype_head, test_loader, criterion, cfg, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "cost_time": time.time() - t0,
        }
        for key, value in val_metrics.items():
            row[f"val_{key}"] = json.dumps(value) if isinstance(value, list) else value
        for key, value in test_metrics.items():
            row[f"test_{key}"] = json.dumps(value) if isinstance(value, list) else value
        rows.append(row)
        auc = float(val_metrics.get("macro_auc", float("-inf")))
        if auc > best_auc:
            best_auc = auc
            best_row = row.copy()
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(model.state_dict(), cfg.output_dir / f"Best_EPOCH_{epoch}.pth")

    torch.save(model.state_dict(), cfg.output_dir / f"Last_EPOCH_{cfg.epochs}.pth")
    log_path = cfg.output_dir / f"Log_seed{cfg.seed}_{cfg.variant}.csv"
    best_path = cfg.output_dir / f"Best_Log_seed{cfg.seed}_{cfg.variant}.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best_row is not None:
        with best_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
            writer.writeheader()
            writer.writerow(best_row)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        for split, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
            write_predictions(
                model,
                dataset,
                output_path=cfg.output_dir / f"{split}_predictions.csv",
                split=split,
                cfg=cfg,
                device=device,
            )
    metadata = {
        "variant": cfg.variant,
        "dataset_csv": str(cfg.dataset_csv),
        "num_classes": cfg.num_classes,
        "in_dim": cfg.in_dim,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "dropout": cfg.dropout,
        "balanced_sampler": cfg.balanced_sampler,
        "focal_gamma": cfg.focal_gamma,
        "focal_beta": cfg.focal_beta,
        "prototype_lambda": cfg.prototype_lambda,
        "prototype_temperature": cfg.prototype_temperature,
        "use_coords": cfg.use_coords,
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train custom AB_MIL variants for Auto-MIL.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--mil-baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        help=(
            "Variant name. Names containing FOCAL use class-balanced focal loss; "
            "names containing PROTO use prototype regularization. Suffixes are kept "
            "for experiment tracking, e.g. AB_MIL_FOCAL_PROTO_LR1E4_PL01."
        ),
    )
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--in-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-beta", type=float, default=0.999)
    parser.add_argument("--prototype-lambda", type=float, default=0.2)
    parser.add_argument("--prototype-temperature", type=float, default=0.2)
    parser.add_argument("--use-coords", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        dataset_csv=Path(args.dataset_csv),
        mil_baseline_dir=assert_mil_baseline_root(args.mil_baseline_dir),
        output_dir=Path(args.output_dir),
        variant=args.variant,
        num_classes=args.num_classes,
        in_dim=args.in_dim + (2 if args.use_coords else 0),
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        lr=args.lr,
        dropout=args.dropout,
        balanced_sampler=args.balanced_sampler,
        focal_gamma=args.focal_gamma,
        focal_beta=args.focal_beta,
        prototype_lambda=args.prototype_lambda,
        prototype_temperature=args.prototype_temperature,
        use_coords=args.use_coords,
    )


def main() -> None:
    cfg = parse_args()
    best_path = train(cfg)
    print(f"best_log={best_path}")


if __name__ == "__main__":
    main()
