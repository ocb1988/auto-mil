# Auto-MIL

Camyla-inspired autonomous research scaffold for multiple-instance learning
(MIL) in computational pathology.

This repository includes a vendored MIL baseline tree under
`third_party/MIL_BASELINE` and orchestrates experiments on top of that bundled
code by default, using slide-level H5 feature bags such as `K:\cptac-brca`.

## What It Does

- Audits a pathology MIL dataset and builds MIL_BASELINE-compatible CSV files.
- Screens several baseline MIL methods under a small budget.
- Ships MIL_BASELINE configs/modules/processors inside the project, including
  classic methods such as `AB_MIL`, `TRANS_MIL`, `RRT_MIL` and recent methods
  such as `STABLE_MIL`, `GDF_MIL`.
- Generates follow-up research recipes from the best baseline results.
- Runs selected recipes through MIL_BASELINE.
- Writes a compact experiment report with dataset, command, metric, and artifact
  provenance.

The design borrows Camyla's staged autonomous research pattern:

1. `dataset_audit`: inspect task/data constraints.
2. `baseline_screen`: run a cheap model sweep.
3. `proposal_queue`: generate ranked research recipes.
4. `focused_runs`: execute promising recipes.
5. `report`: summarize results and next hypotheses.

## Quick Demo

Use the Torch environment you specified:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli prepare-cptac `
  --data-dir K:\cptac-brca `
  --output-dir runs\cptac_brca_pam50
```

Then run a small autonomous loop:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run `
  --config configs\cptac_brca_pam50.yaml `
  --max-screen-runs 3 `
  --max-focused-runs 2
```

Resume an interrupted run from its checkpoint:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run-cv `
  --config configs\cptac_brca_pam50.yaml `
  --resume
```

Checkpoint files are written as `checkpoint.json` plus
`checkpoint_events.jsonl` inside each run directory. Resume reuses only runs
marked `completed`; failed and dry-run entries are rerun.

For a non-training smoke test:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run `
  --config configs\cptac_brca_pam50.yaml `
  --dry-run
```

List the bundled baseline methods:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli list-baselines --trainable-only
```

Inspect a checkpoint:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli checkpoint-summary `
  --path runs\cptac_brca_pam50\case_level_cv
```

Classify a failed run log:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli analyze-log `
  --path runs\cptac_brca_pam50\case_level_cv\stdout
```

Failed runs also write a sibling `*.diagnosis.json` next to the stdout log and
store the diagnosis in `checkpoint.json`.

Print the conservative retry/escalation policy for one failed log:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli failure-action `
  --path runs\cptac_brca_pam50\case_level_cv\stdout\my_failed_run.log
```

The failure policy can plan safe retry overrides for categories such as CUDA
OOM, NaN loss, timeout, or interruption. It deliberately pauses for dependency,
data, shape, H5-key, or split-policy problems instead of mutating the
environment or dataset automatically.

Run a QWBE-lite experiment tree search:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run-tree `
  --config configs\cptac_brca_pam50.yaml `
  --max-runs 6 `
  --resume
```

This writes `experiment_tree.json`, `experiment_tree.md`, a checkpoint, and raw
MIL_BASELINE artifacts under `runs/cptac_brca_pam50/experiment_tree/`. Failed
tree nodes are diagnosed; retryable failures can add one conservative retry
child node within the configured failure-retry depth.

## Outputs

By default, outputs land under `runs/cptac_brca_pam50/`:

- `dataset.csv`: MIL_BASELINE train/val/test wide CSV.
- `metadata.json`: label mapping, split counts, inferred feature dimension.
- `configs/*.yaml`: generated MIL_BASELINE configs.
- `mil_logs/`: raw MIL_BASELINE training outputs.
- `research_journal.jsonl`: append-only stage journal.
- `report.md`: autonomous research summary.

## Manuscript Experiment Skill

This repo includes a Codex skill for manuscript-grade pathology MIL experiment
workflows:

```text
skills/pathology-mil-experiment/
```

Use it when planning the experimental section of a pathology AI paper. It guides
task clarification, runtime checks, data exploration, split design, baseline
selection, autonomous model innovation, ablation studies, experiment/git
logging, and Methods/Results drafting.
