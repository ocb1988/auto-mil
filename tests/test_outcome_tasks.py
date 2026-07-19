from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from auto_mil.outcome_tasks import VENDORED_OUTCOME_MODELS, OutcomeMIL, concordance_index, prepare_outcome_kfold
from auto_mil.specs import DatasetSpec, FeatureSpec, TaskSpec


def _write_feature_dataset(tmp_path: Path, labels: pd.DataFrame) -> DatasetSpec:
    data_dir = tmp_path / "features"
    data_dir.mkdir()
    for index, case_id in enumerate(labels["case_id"]):
        torch.save({"features": torch.full((3 + index % 2, 4), float(index))}, data_dir / f"{case_id}-slide.pt")
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    return DatasetSpec(
        name="synthetic-outcome",
        data_dir=data_dir,
        labels_csv=labels_path,
        case_id_column="case_id",
        feature=FeatureSpec(format="pt", feature_glob="*.pt", case_id_regex=r"^(?P<case>case_\d+)-"),
    )


class OutcomeTaskTests(unittest.TestCase):
    def test_regression_kfold_creates_case_level_bags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            labels = pd.DataFrame(
                {
                    "case_id": [f"case_{index}" for index in range(12)],
                    "target": [float(index) for index in range(12)],
                }
            )
            dataset = _write_feature_dataset(tmp_path, labels)
            task = TaskSpec(kind="regression", target_column="target", split_seed=7)
            artifacts = prepare_outcome_kfold(dataset=dataset, task=task, output_dir=tmp_path / "prepared", n_splits=3)

            self.assertEqual(len(artifacts.fold_dirs), 3)
            metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["num_cases"], 12)
            self.assertEqual(metadata["bag_level"], "case")
            for fold_dir in artifacts.fold_dirs:
                bags = pd.read_csv(fold_dir / "bags.csv")
                self.assertEqual(set(bags["split"]), {"train", "val", "test"})
                self.assertEqual(len(bags), 12)
                self.assertTrue(all((fold_dir / "case_bags" / f"{case_id}.h5").exists() for case_id in bags["case_id"]))

    def test_survival_concordance_index_handles_ties_and_censoring(self) -> None:
        self.assertEqual(concordance_index([5.0, 4.0, 3.0], [1, 1, 0], [0.1, 0.5, 0.9]), 1.0)
        self.assertEqual(concordance_index([5.0, 4.0], [1, 1], [0.5, 0.5]), 0.5)

    def test_survival_kfold_accepts_time_and_event_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            labels = pd.DataFrame(
                {
                    "case_id": [f"case_{index}" for index in range(12)],
                    "time": [float(index + 1) for index in range(12)],
                    "event": [index % 2 for index in range(12)],
                }
            )
            dataset = _write_feature_dataset(tmp_path, labels)
            task = TaskSpec(kind="prognosis", time_column="time", event_column="event", split_seed=11)
            artifacts = prepare_outcome_kfold(dataset=dataset, task=task, output_dir=tmp_path / "prepared", n_splits=3)

            metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["survival"]["events"], 6)
            self.assertEqual(len(artifacts.fold_dirs), 3)

    def test_vendored_aggregators_accept_task_specific_heads(self) -> None:
        for model_name in ["AB_MIL", "TRANS_MIL", "RRT_MIL", "STABLE_MIL", "GDF_MIL"]:
            model = OutcomeMIL(in_dim=8, model_name=model_name, dropout=0.0)
            prediction = model(torch.randn(1, 9, 8))
            self.assertEqual(tuple(prediction.shape), (1,), model_name)
            prediction.sum().backward()
            self.assertIsNotNone(model.task_head.weight.grad, model_name)

    def test_generic_outcome_adapter_covers_extra_backbone_interfaces(self) -> None:
        self.assertEqual(len(VENDORED_OUTCOME_MODELS), 41)
        for model_name in [
            "AC_MIL",
            "CDP_MIL",
            "MICO_MIL",
            "NCIE_MIL",
            "PGCN_MIL",
            "RET_MIL",
            "S4_MIL",
            "SC_MIL",
            "WIKG_MIL",
        ]:
            model = OutcomeMIL(in_dim=8, model_name=model_name, dropout=0.0)
            prediction = model(torch.randn(1, 12, 8), torch.randn(1, 12, 2))
            self.assertEqual(tuple(prediction.shape), (1,), model_name)
            prediction.sum().backward()
            self.assertIsNotNone(model.task_head.weight.grad, model_name)
