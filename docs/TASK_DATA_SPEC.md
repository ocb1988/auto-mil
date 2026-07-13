# Task and Data Specification

Auto-MIL normalizes user/project configuration into two stable interfaces:

- `TaskSpec`: the prediction target and evaluation contract
- `DatasetSpec`: where bags, labels, identifiers, cohorts, and feature keys live

The current training adapter executes classification tasks with H5 feature bags
through MIL_BASELINE. Prognosis/survival and regression fields are already
represented in `TaskSpec` so future adapters can be added without changing the
outer research workflow.

## Config Shape

```yaml
dataset:
  name: CPTAC-BRCA
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

## Supported Now

- task kind: `classification`
- feature format: H5 files under `dataset.data_dir` or `paths.data_dir`
- feature key: configurable, defaults to `features`
- split unit: case/patient id
- labels: CSV with a case id column and a classification label column
- optional metadata: center, cohort, external-test indicator, coordinate key
- split planning: predefined external-test, center-holdout, single-cohort
  stratified CV, and pilot train/val/test holdout

## Reserved Next

- `prognosis` / `survival`: `time_column` and `event_column`
- `regression`: `target_column`
- execution adapters that consume a confirmed split plan directly
- explicit slide manifest files through `slide_path_column`
