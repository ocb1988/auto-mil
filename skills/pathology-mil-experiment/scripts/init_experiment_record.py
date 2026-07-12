from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a pathology MIL experiment record.")
    parser.add_argument("--run-dir", required=True, help="Experiment run directory.")
    parser.add_argument("--title", required=True, help="Experiment title.")
    parser.add_argument("--task", default="", help="Optional task summary.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "git_snapshots").mkdir(exist_ok=True)

    plan = run_dir / "experiment_plan.md"
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

    journal = run_dir / "experiment_journal.jsonl"
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
    print(f"git_snapshots={run_dir / 'git_snapshots'}")


if __name__ == "__main__":
    main()
