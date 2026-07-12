from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .baseline_registry import resolve_mil_baseline_dir


@dataclass(frozen=True)
class AutoMilConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def name(self) -> str:
        return str(self.raw.get("name", self.path.stem))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["paths"]["output_dir"])

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["paths"]["data_dir"])

    @property
    def labels_csv(self) -> Path:
        return Path(self.raw["paths"]["labels_csv"])

    @property
    def mil_baseline_dir(self) -> Path:
        return resolve_mil_baseline_dir(self.raw.get("paths", {}).get("mil_baseline_dir"))

    @property
    def python(self) -> Path:
        return Path(self.raw["paths"]["python"])


def load_config(path: str | Path) -> AutoMilConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AutoMilConfig(raw=raw, path=config_path)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
