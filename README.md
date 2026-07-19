# Auto-MIL

Camyla-inspired autonomous research scaffold for multiple-instance learning
(MIL) in computational pathology.

This repository includes a vendored MIL baseline tree under
`third_party/MIL_BASELINE` and orchestrates experiments on top of that bundled
code by default, using H5 or PT feature bags such as `K:\cptac-brca` or a
recursive WSI feature directory.

## What It Does

- Audits a pathology MIL dataset and builds MIL_BASELINE-compatible CSV files.
- Builds patient/case-level bags by concatenating patch features from all slides
  belonging to the same case by default.
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

Prepare a generic configured dataset without launching training:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli prepare-data `
  --config configs\gastric_ki67_binary50.yaml
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

After CV finishes, compute statistical summaries:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli analyze-stats `
  --checkpoint runs\cptac_brca_pam50\case_level_cv\checkpoint.json `
  --metric test_macro_auc `
  --baseline AB_MIL
```

Collect all experiment checkpoints into manuscript-ready result tables:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli collect-results `
  --root runs\cptac_brca_pam50 `
  --primary-metric test_macro_auc
```

Aggregate slide-level prediction CSVs to patient/case level:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli aggregate-predictions `
  --root runs\cptac_brca_pam50 `
  --aggregation mean
```

Build manuscript figures and error-case tables from case-level predictions:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli build-figures `
  --root runs\cptac_brca_pam50
```

Draft manuscript Methods/Results from the collected evidence artifacts:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli write-manuscript `
  --root runs\cptac_brca_pam50 `
  --config configs\cptac_brca_pam50.yaml
```

Package the draft for polishing and submission checks:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli package-manuscript `
  --root runs\cptac_brca_pam50 `
  --profile generic-pathology-ai `
  --target-journal "target journal name"
```

Run the default innovation ablation matrix:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run-ablation-cv `
  --config configs\cptac_brca_pam50.yaml `
  --split-plan runs\cptac_brca_pam50\split_plan\split_plan.json
```

Run an approved time-boxed autonomous window:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli run-autonomous-window `
  --config configs\cptac_brca_pam50.yaml `
  --max-minutes 120 `
  --max-runs 20 `
  --target-metric test_macro_auc `
  --target-value 0.85 `
  --split-plan runs\cptac_brca_pam50\split_plan\split_plan.json `
  --plan-id patient_stratified_holdout `
  --resume
```

Preview or add new experiment-tree proposals from current evidence:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli propose-nodes `
  --config configs\cptac_brca_pam50.yaml `
  --max-proposals 6
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

- `dataset.csv`: MIL_BASELINE train/val/test wide CSV. By default paths point
  to generated patient/case-level H5 bags under `case_bags/`.
- `bags_long.csv` and `case_bag_manifest.csv`: case-bag inventory and source
  slide provenance when `dataset.bag_level: case`.
- `metadata.json`: label mapping, split counts, inferred feature dimension.
- `configs/*.yaml`: generated MIL_BASELINE configs.
- `mil_logs/`: raw MIL_BASELINE training outputs.
- `research_journal.jsonl`: append-only stage journal.
- `report.md`: autonomous research summary.
- `split_plan/split_plan.json` and `split_plan/split_plan.md`: proposed split
  options and confirmation record.

## Task/Data Interface

Configuration is normalized into `TaskSpec` and `DatasetSpec`. The runner
executes classification, regression, and prognosis/survival with H5 feature
bags or PT feature dictionaries containing configurable feature keys. The
classification path uses MIL_BASELINE; regression and survival use the built-in
task-aware adapter. For outcome tasks it supports all 41 audited head/loss
vendored backbones, reusing their encoder/aggregator and replacing the
classification head with a one-output regression/risk head. The bundled starter
screen is `AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL`;
`MEAN_MIL`, `MAX_MIL`, and `GATE_AB_MIL` remain lightweight sanity baselines.
Select the final five comparison methods through task-directed literature
evidence and runtime validation. See
`docs/VENDORED_OUTCOME_ADAPTATION_AUDIT.md` for constraints and dependencies.
The default bag level is `case`, meaning multiple slides from the same
patient/case are concatenated into one MIL bag before training. Set
`dataset.bag_level: slide` only when intentionally running slide-level MIL.
Regression reports MAE, RMSE, R2, and Spearman correlation. Survival uses Cox
partial likelihood and reports C-index. See `docs/TASK_DATA_SPEC.md`.

## Split Planner

`plan-split` inspects the matched case table and proposes manuscript-grade split
options before any baseline run. It prioritizes predefined external tests,
center-aware holdout when center fields exist, and patient-level stratified
n-fold CV for single-cohort datasets.

Training commands can consume a confirmed plan:

- `run-cv --split-plan ...` accepts `n_fold_cross_validation` plans.
- `run-innovation-cv --split-plan ...` uses the same CV split for innovation experiments.
- `run --split-plan ... --plan-id patient_stratified_holdout` accepts holdout/external/center plans for classification, regression, and survival.
- `run-tree --split-plan ... --plan-id ...` accepts holdout/external/center plans.

The selected plan is written into run metadata under `confirmed_split`.

## Baseline Planner

`plan-baselines` checks the bundled MIL_BASELINE methods against the current
dataset and runtime. It reports trainability, baseline family, coordinate needs,
dependency status, memory risk, and a recommended **functional starter screen**.
The bundled default (`AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and
`GDF_MIL` when compatible) validates the pipeline but is not an automatic
manuscript comparison table. Before final comparison, use task-directed literature
evidence to select exactly five task-valid methods, prioritizing work from the
past 1-2 years in high-impact journals or leading conferences, then confirm that
each method is reproducible in the active runtime.

## Statistical Analysis

`analyze-stats` reads completed fold/seed runs from `checkpoint.json` and writes
`stats_report.json/md` with mean, standard deviation, 95% confidence intervals,
and paired comparisons against a chosen baseline when common folds exist.

## Result Collector

`collect-results` discovers `checkpoint.json` files under a run root and writes a
stable result bundle for manuscript drafting: `results_index.json`, `runs.csv`,
`model_summary.csv`, and `manuscript_results.md`. It can include failed records
for auditability or use `--completed-only` for clean result tables.

## Prediction Aggregation

`aggregate-predictions` is an optional post-hoc path for experiments that emit
slide-level predictions. It aggregates multiple slides from the same
patient/case into case-level probabilities using mean, median, or max pooling
and writes `slide_predictions.csv`, `case_predictions.csv`, `case_metrics.json`,
and `prediction_report.md`. The custom AB_MIL innovation runner writes
`train_predictions.csv`, `val_predictions.csv`, and `test_predictions.csv`
automatically after non-dry-run training.

## Figure Report

`build-figures` reads `case_predictions.csv` and writes manuscript-facing
artifacts: `roc_curve.png`, `confusion_matrix.png`, `calibration_curve.png`
when binary probabilities are available, `per_class_metrics.csv`,
`error_cases.csv`, and `figure_report.md/json`.

## Manuscript Writer

`write-manuscript` reads the available evidence artifacts under a run root and
writes `manuscript/manuscript_evidence.json` plus
`manuscript/manuscript_draft.md`. It uses metadata, split plans, result tables,
statistical reports, prediction reports, figure reports, and ablation reports
when present, and emits warnings for missing or dry-run-only evidence. The draft
is intentionally conservative: it scaffolds Methods/Results text and claim
guardrails without inventing unsupported performance claims.

## Manuscript Package

`package-manuscript` turns the draft and evidence manifest into a polishing
work package under `manuscript/package/`: `methods_results_package.md`,
`llm_polish_prompt.md`, `submission_checklist.md`, and
`manuscript_package.json`. Profiles include `generic-pathology-ai`,
`structured-medical-ai`, and `methods-results-only`. A named target journal is
treated as a free-text style label; current author instructions should still be
checked before submission.

## Ablation Runner

`run-ablation-cv` runs a fixed AB_MIL innovation matrix on the confirmed CV
split: cross-entropy baseline, focal loss only, prototype head only, and the
full focal-plus-prototype method. It writes `ablation_cv_report.md` and a
checkpoint for resume.

## Literature and Proposal Search

Before starting a method-innovation loop, run one literature/proposal search.
If the user provides a local method-review corpus, use it before online search:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli literature-search `
  --config configs\gastric_ki67_binary50.yaml `
  --local-review MIL_methods_overview.md `
  --offline
```

If the user provides papers or proposal drafts, pass them as JSON/CSV/TSV and
use `--offline` to avoid online search:

```powershell
& "D:\ProgramData\Anaconda3\envs\torch2p7cu128\python.exe" -m auto_mil.cli literature-search `
  --config configs\gastric_ki67_binary50.yaml `
  --user-sources papers.json `
  --offline
```

If no sources are provided, omit `--user-sources`; the command queries online
paper indexes and writes `literature_search/literature_report.md` plus
`literature_search/literature_proposals.json`. `propose-nodes` can read this
JSON as method proposal seeds when `innovation.method_proposals` is absent.

## Autonomous Window

`run-autonomous-window` runs one QWBE-lite tree node per round until the approved
wall-clock budget, maximum run count, or target metric is reached. It records an
`autonomous_journal.jsonl` and `autonomous_summary.md`, uses the confirmed split
plan, respects per-run timeout if provided, and relies on failure policy for
safe retries. When pending nodes are exhausted, it can call the proposal
generator to add new candidate nodes from the current evidence. If a support-only
node such as local tuning reaches the target, the window records that event but
continues method-track exploration until a method node reaches the target or the
approved budget is exhausted.

## Proposal Generator

`propose-nodes` reads the current experiment tree, checkpoint, and
`baseline_plan.json`, then adds auditable `ExperimentNode` proposals. The current
deterministic generator proposes compatible baseline-family expansions and local
refinements of the best completed recipe. It also reads optional
`innovation.method_proposals` entries from the config for manuscript-worthy
method proposals with explicit `core_modules`, `support_tags`, and
`ablation_plan`. Use `--no-apply` to preview without modifying the tree.

```yaml
innovation:
  method_proposals:
    - name: focal_proto_abmil
      model_name: AB_MIL
      core_modules:
        - class-balanced focal objective
        - prototype auxiliary head
      ablation_plan:
        - remove focal objective
        - remove prototype auxiliary head
      config_overrides: {}
```

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
