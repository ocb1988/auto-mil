# Task and Data Specification

Auto-MIL normalizes user/project configuration into two stable interfaces:

- `TaskSpec`: the prediction target and evaluation contract
- `DatasetSpec`: where bags, labels, identifiers, cohorts, and feature keys live

Classification tasks execute through MIL_BASELINE. Regression and
prognosis/survival execute through the built-in task-aware MIL adapter with
task-specific heads/losses. Their initial screen can reuse vendored `AB_MIL`,
`TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL` aggregators, while keeping
mean/max/gated-attention MIL as sanity baselines. All three task kinds accept H5
or PT feature bags and preserve the same case-level workflow.

## Config Shape

```yaml
dataset:
  name: CPTAC-BRCA
  bag_level: case              # default; use slide only for slide-level MIL
  case_id_column: case_id
  center_column: center        # optional
  cohort_column: cohort        # optional
  external_test_column: split  # optional
  feature:
    format: h5
    feature_key: features
    coords_key: coords         # optional
    case_id_regex: "^(?P<case>[A-Za-z0-9]+)-"

task:
  kind: classification
  label_column: PAM50
  primary_metric: test_macro_auc
  min_class_count: 2
  split_seed: 2024
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  cv_val_fraction_of_train: 0.2
```

`paths.data_dir` and `paths.labels_csv` remain supported for backward
compatibility. `dataset.data_dir` and `dataset.labels_csv` may be added when a
project wants all dataset fields grouped together.

## Inspection

Before running experiments, normalize and inspect the config:

```powershell
python -m auto_mil.cli inspect-spec --config configs\cptac_brca_pam50.yaml
```

Use `--json` to capture the normalized task/data payload in experiment logs.

## Split Planning

After inspection, propose split options:

```powershell
python -m auto_mil.cli plan-split --config configs\cptac_brca_pam50.yaml
```

This writes `split_plan.json` and `split_plan.md` under
`runs/<experiment>/split_plan/` by default. The planner reports case/slide/class
counts, optional center/cohort/external-test distributions, recommended plans,
warnings, and a confirmation gate.

After confirmation, pass the plan to execution:

```powershell
python -m auto_mil.cli run-cv --config configs\cptac_brca_pam50.yaml --split-plan runs\cptac_brca_pam50\split_plan\split_plan.json
```

Use `--plan-id` when multiple plans are recommended or when selecting a
non-default holdout/pilot plan. The selected plan is stored in `metadata.json`
as `confirmed_split`.

## Baseline Planning

After locking the split, assess candidate baselines:

```powershell
python -m auto_mil.cli plan-baselines --config configs\cptac_brca_pam50.yaml
```

The planner inspects feature keys and coordinate availability, then writes
`baseline_plan.json` and `baseline_plan.md` under
`runs/<experiment>/baseline_plan/`.

## Supported Now

- task kinds: `classification`, `regression`, `prognosis` / `survival`
- feature format: H5 or PT files under `dataset.data_dir` or `paths.data_dir`
- feature key: configurable, defaults to `features`
- split unit: case/patient id
- bag level: `case` by default. Auto-MIL concatenates patch features from all
  slides belonging to the same case into one generated H5 bag before training.
  Use `dataset.bag_level: slide` only when deliberately treating each slide as a
  separate MIL sample.
- classification labels: `task.label_column`, with optional `label_threshold`
- regression labels: numeric `task.target_column`; reports MAE/RMSE/R2/Spearman
- survival labels: positive `task.time_column` and binary `task.event_column`; trains with Cox partial likelihood and reports C-index
- outcome MIL backbones: all 41 audited vendored head/loss candidates, which
  retain their patch encoder/aggregator but replace the classification head with
  one continuous risk/target head. The bundled functional starter screen is
  `AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL`; select the
  final five comparison methods through task-directed literature evidence and
  runtime validation. The full compatibility, coordinate, asset, and dependency
  matrix is in
  `docs/VENDORED_OUTCOME_ADAPTATION_AUDIT.md`. Real patch coordinates remain
  preferable for spatial methods, even when a pseudo-grid fallback exists.
- optional metadata: center, cohort, external-test indicator, coordinate key
- split planning: predefined external-test, center-holdout, single-cohort
  stratified CV, and pilot train/val/test holdout
- split execution: `run-cv` executes deterministic patient-level CV for all
  three task kinds; `run` executes confirmed patient-level holdout, external
  test, or center-holdout plans for all three task kinds. Both record the
  confirmed plan in output metadata. The current innovation/ablation/tree
  paths remain classification-only.

## Task Examples

```yaml
task:
  kind: regression
  target_column: ki67_percent
  primary_metric: test_rmse
```

```yaml
task:
  kind: survival
  time_column: overall_survival_months
  event_column: overall_survival_event
  primary_metric: test_c_index
```

Use `run-cv --models AB_MIL,TRANS_MIL,RRT_MIL,STABLE_MIL,GDF_MIL` only for the
first outcome-task functional screen; add `MEAN_MIL`, `MAX_MIL`, or
`GATE_AB_MIL` as sanity baselines. Before reporting a manuscript comparison,
replace that list with the five literature-selected methods that are directly
valid for the target task and reproducible in the active runtime. Explicit slide
manifest files through `slide_path_column` are also supported.
