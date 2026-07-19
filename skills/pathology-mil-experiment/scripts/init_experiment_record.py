from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _git_snapshot() -> dict[str, object]:
    def run_git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a pathology MIL experiment record.")
    parser.add_argument("--run-dir", required=True, help="Experiment run directory.")
    parser.add_argument("--title", required=True, help="Experiment title.")
    parser.add_argument("--task", default="", help="Optional task summary.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    project_dir = run_dir / "project"
    project_dir.mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    for directory in ("configs", "git_snapshots", "patches", "records", "src"):
        (project_dir / directory).mkdir(exist_ok=True)

    source_init = project_dir / "src" / "__init__.py"
    if not source_init.exists():
        source_init.write_text(
            '"""Project-specific Auto-MIL extensions. Keep reusable core unchanged."""\n',
            encoding="utf-8",
        )

    manifest = project_dir / "project_manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "title": args.title,
                    "task": args.task,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "workspace_policy": {
                        "core_immutable_for_project_work": True,
                        "project_source_dir": "src",
                        "project_config_dir": "configs",
                        "project_patch_dir": "patches",
                        "artifact_dir": "artifacts",
                    },
                    "core_git": _git_snapshot(),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    plan = project_dir / "experiment_plan.md"
    if not plan.exists():
        plan.write_text(
            "\n".join(
                [
                    f"# {args.title}",
                    "",
                    f"Created: {datetime.now().isoformat(timespec='seconds')}",
                    "",
                    "## Task",
                    "",
                    args.task or "TBD",
                    "",
                    "## Workspace Isolation",
                    "",
                    f"- Project workspace: `{project_dir}`",
                    "- Reusable core: do not edit `auto_mil/` or `third_party/MIL_BASELINE/` for this project.",
                    "- Project code: `src/`",
                    "- Project configs: `configs/`",
                    "- Adapter/override patches: `patches/`",
                    "- Core git revision: see `project_manifest.json`.",
                    "",
                    "## Data Exploration",
                    "",
                    "- Data path:",
                    "- Label file:",
                    "- Endpoint:",
                    "- Prediction unit:",
                    "- Counts:",
                    "- Missingness:",
                    "- Leakage risks:",
                    "",
                    "## Runtime",
                    "",
                    "- Local or remote:",
                    "- Python:",
                    "- Conda/pip:",
                    "- CUDA:",
                    "- Torch:",
                    "- GPU:",
                    "",
                    "## Split Plan",
                    "",
                    "- Proposed split:",
                    "- Rationale:",
                    "- User confirmation:",
                    "",
                    "## Baselines",
                    "",
                    "| Method | Config | Status | Primary metric | Notes |",
                    "|---|---|---|---:|---|",
                    "",
                    "## Innovation Loop",
                    "",
                    "| Round | Idea | Command | Result | Reflection | Decision |",
                    "|---:|---|---|---|---|---|",
                    "",
                    "## Ablations",
                    "",
                    "| Variant | Change | Primary metric | Interpretation |",
                    "|---|---|---:|---|",
                    "",
                    "## Manuscript Claims",
                    "",
                    "- TBD",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    journal = project_dir / "experiment_journal.jsonl"
    if not journal.exists():
        first = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": "init",
            "title": args.title,
            "task": args.task,
        }
        journal.write_text(json.dumps(first, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"plan={plan}")
    print(f"journal={journal}")
    print(f"manifest={manifest}")
    print(f"project_source={project_dir / 'src'}")
    print(f"git_snapshots={project_dir / 'git_snapshots'}")


if __name__ == "__main__":
    main()
