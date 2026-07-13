# Auto-MIL

Camyla-inspired autonomous research scaffold for multiple-instance learning
(MIL) in computational pathology.

This repository includes a vendored MIL baseline tree under
`third_party/MIL_BASELINE` and orchestrates experiments on top of that bundled
code by default, using slide-level H5 feature bags such as `K:\cptac-brca`.

## What It Does

- Audits a pathology MIL dataset and builds MIL_BASELINE-compatible CSV files.
- Normalizes each project into explicit `TaskSpec` and `DatasetSpec` records.
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

Inspect the normalized task/data interface:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli inspect-spec `
  --config configs\cptac_brca_pam50.yaml
```

Generate a split plan and stop for confirmation:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli plan-split `
  --config configs\cptac_brca_pam50.yaml
```

After confirming the plan, pass it into execution so the split is locked:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run-cv `
  --config configs\cptac_brca_pam50.yaml `
  --split-plan runs\cptac_brca_pam50\split_plan\split_plan.json
```

Assess baseline families before launching the screen:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli plan-baselines `
  --config configs\cptac_brca_pam50.yaml
```

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
- `split_plan/split_plan.json` and `split_plan/split_plan.md`: proposed split
  options and confirmation record.

## Task/Data Interface

Configuration is normalized into `TaskSpec` and `DatasetSpec`. The current
runner supports classification with H5 feature bags and configurable feature
keys; prognosis/survival and regression fields are represented for the next
adapters. See `docs/TASK_DATA_SPEC.md`.

## Split Planner

`plan-split` inspects the matched case table and proposes manuscript-grade split
options before any baseline run. It prioritizes predefined external tests,
center-aware holdout when center fields exist, and patient-level stratified
n-fold CV for single-cohort datasets.

Training commands can consume a confirmed plan:

- `run-cv --split-plan ...` accepts `n_fold_cross_validation` plans.
- `run-innovation-cv --split-plan ...` uses the same CV split for innovation experiments.
- `run --split-plan ... --plan-id patient_stratified_holdout` accepts holdout/external/center plans.
- `run-tree --split-plan ... --plan-id ...` accepts holdout/external/center plans.

The selected plan is written into run metadata under `confirmed_split`.

## Baseline Planner

`plan-baselines` checks the bundled MIL_BASELINE methods against the current
dataset and runtime. It reports trainability, baseline family, coordinate needs,
dependency status, memory risk, and a recommended manuscript screen. The default
screen is `AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL` when
compatible.

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
