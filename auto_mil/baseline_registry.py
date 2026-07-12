from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_MIL_BASELINE_DIR = PROJECT_ROOT / "third_party" / "MIL_BASELINE"


@dataclass(frozen=True)
class ModelAvailability:
    model_name: str
    config_path: Path
    module_dir: Path | None
    process_dir: Path | None
    has_config: bool
    has_module: bool
    has_process: bool

    @property
    def is_trainable(self) -> bool:
        return self.has_config and self.has_module and self.has_process


def resolve_mil_baseline_dir(value: str | Path | None = None) -> Path:
    if value is None:
        return BUNDLED_MIL_BASELINE_DIR
    text = str(value).strip()
    if text.lower() in {"", "auto", "bundled", "internal", "third_party"}:
        return BUNDLED_MIL_BASELINE_DIR
    path = Path(text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def assert_mil_baseline_root(root: str | Path | None = None) -> Path:
    root_path = resolve_mil_baseline_dir(root)
    required = [
        root_path / "train_mil.py",
        root_path / "configs",
        root_path / "modules",
        root_path / "process",
        root_path / "utils",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"MIL_BASELINE root is incomplete: {root_path}\nMissing:\n{joined}")
    return root_path


def available_models(root: str | Path | None = None) -> dict[str, ModelAvailability]:
    root_path = assert_mil_baseline_root(root)
    config_names = {path.stem for path in (root_path / "configs").glob("*.yaml")}
    module_names = {path.name for path in (root_path / "modules").iterdir() if path.is_dir()}
    process_names = {path.name for path in (root_path / "process").iterdir() if path.is_dir()}
    names = sorted(config_names | module_names | process_names)
    out: dict[str, ModelAvailability] = {}
    for name in names:
        module_dir = root_path / "modules" / name
        process_dir = root_path / "process" / name
        out[name] = ModelAvailability(
            model_name=name,
            config_path=root_path / "configs" / f"{name}.yaml",
            module_dir=module_dir if module_dir.exists() else None,
            process_dir=process_dir if process_dir.exists() else None,
            has_config=name in config_names,
            has_module=name in module_names,
            has_process=name in process_names,
        )
    return out


def assert_models_available(models: list[str], root: str | Path | None = None) -> None:
    registry = available_models(root)
    missing: list[str] = []
    incomplete: list[str] = []
    for model in models:
        info = registry.get(model)
        if info is None:
            missing.append(model)
        elif not info.is_trainable:
            parts = []
            if not info.has_config:
                parts.append("config")
            if not info.has_module:
                parts.append("module")
            if not info.has_process:
                parts.append("process")
            incomplete.append(f"{model} missing {', '.join(parts)}")
    if missing or incomplete:
        lines = []
        if missing:
            lines.append("Missing models: " + ", ".join(missing))
        lines.extend(incomplete)
        raise FileNotFoundError("Unavailable MIL baseline model(s):\n" + "\n".join(lines))
