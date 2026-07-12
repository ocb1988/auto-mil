from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


class ResearchJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, payload: dict[str, Any]) -> None:
        record = {"time": now_iso(), "stage": stage, "payload": json_ready(payload)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


class ExperimentCheckpoint:
    def __init__(self, path: Path):
        self.path = path
        self.events_path = path.with_name(path.stem + "_events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "metadata": {},
                "runs": {},
            }
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        self.state["updated_at"] = now_iso()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=True)
        tmp_path.replace(self.path)

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {"time": now_iso(), "event": event, "payload": json_ready(payload)}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def update_metadata(self, **metadata: Any) -> None:
        self.state.setdefault("metadata", {}).update(json_ready(metadata))
        self._save()
        self._write_event("metadata", metadata)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.state.setdefault("runs", {}).get(run_id)
        return run if isinstance(run, dict) else None

    def get_completed_payload(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run and run.get("status") == "completed":
            payload = run.get("payload")
            return payload if isinstance(payload, dict) else None
        return None

    def record_run(self, run_id: str, stage: str, status: str, payload: dict[str, Any]) -> None:
        record = {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "updated_at": now_iso(),
            "payload": json_ready(payload),
        }
        self.state.setdefault("runs", {})[run_id] = record
        self._save()
        self._write_event("run", record)
