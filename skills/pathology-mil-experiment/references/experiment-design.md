# Experiment Design Reference

## Initial Clarification

Collect:

- data path(s), label/clinical file path, feature file type (`.h5`, `.pt`, raw WSI)
- task endpoint: classification, prognosis/survival, regression, detection, retrieval, or multimodal
- unit of prediction: slide, patient/case, specimen, ROI
- cohort/center/site/stain/scanner fields
- target manuscript venue expectations if known
- runtime: local/remote, Python, conda/pip, CUDA, torch, GPU memory

## Data Exploration

Minimum exploration:

- number of patients/cases, slides, centers, labels
- slides per patient and whether labels are patient-level or slide-level
- bag construction policy. For patient/case-level labels, default to one MIL bag
  per case by concatenating patch features from all slides before training.
  Treat slide-level output average/max/min as a fallback or secondary analysis,
  not the default training design.
- class/event/outcome distribution by center
- missing labels, duplicate IDs, inconsistent IDs
- feature dimensions, patch counts, coordinate availability
- train/val/test leakage risks

For H5 or PT feature bags, inspect representative files:

- keys
- `features` shape and dtype
- coordinates shape if present
- masks/stitch thumbnails if present

For numeric biomarker or IHC-style labels, confirm whether the study should be
regression or thresholded classification. If using a threshold for a smoke
test, record the exact rule and avoid treating that pilot threshold as a final
clinical claim until the user confirms the endpoint definition.

## Split Design

Use this priority order:

1. Multi-center or multi-dataset external validation:
   - Train/validation on development center(s)
   - External test on held-out center(s)
   - If centers are imbalanced, report center-level counts and limitations

2. Single dataset:
   - Patient-level k-fold cross validation, usually 5-fold
   - Stratify by label or outcome bins where possible
   - For small datasets, repeat CV across seeds if runtime allows

3. Fixed train/val/test split:
   - Use only when required by benchmark, prior work, or user constraints
   - Split by patient/case
   - Preserve a locked test set

When using this repo, run the split planner before confirming baselines:

```powershell
python -m auto_mil.cli plan-split --config configs\my_experiment.yaml
```

Review `split_plan.md` with the user. Confirm one plan before running
manuscript-grade baselines, and do not change the split after inspecting test
performance.

After confirmation, execute with the confirmed plan:

```powershell
python -m auto_mil.cli run-cv --config configs\my_experiment.yaml --split-plan runs\my_experiment\split_plan\split_plan.json
```

Pass `--plan-id` explicitly when multiple plans are recommended or when the
user chooses a non-default holdout plan.

Survival:

- preserve event/censor distribution
- use C-index and time-dependent AUC when feasible
- avoid random slide-level splitting

Regression:

- stratify by outcome quantiles when feasible
- report MAE/RMSE/R2 plus calibration or rank correlation when relevant

## Baseline Selection

Use a two-stage baseline policy. First run a small functional screen to validate
the pipeline; `AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL`,
with optional `MEAN_MIL`/`MAX_MIL`, are suitable bundled starter models. They
are not an automatic manuscript comparison suite.

For the final table, select exactly five methods after a task-directed
literature search. Prioritize work from the preceding 1-2 years in high-impact
journals (IEEE-TMI, Medical Image Analysis, IEEE-TPAMI) or leading conferences
(CVPR, NeurIPS, AAAI, ICLR, ICML), then filter for direct classification,
regression, or survival validity. Record each candidate's paper, year, venue,
task evidence, repo/code mapping, input requirements, reproducibility status,
and exclusion reason. Use older classic methods only as clearly labeled
historical anchors when fewer than five recent, reproducible, task-valid methods
are available.

Before confirming the baseline suite:

- verify each method has a config and implementation in the target repo
- check special requirements such as coordinates, graph construction, patch count limits, GPU memory, and package dependencies
- state which methods are selected for the pilot screen and which are reserved for the final manuscript-scale table

When using this repo, run the baseline planner after confirming the split:

```powershell
python -m auto_mil.cli plan-baselines --config configs\my_experiment.yaml
```

Review `baseline_plan.md` with the user before training. Treat its default
screen as a runtime compatibility check, then present the literature evidence
table and obtain confirmation for the five final comparison methods. Add spatial
methods such as `DAG_MIL`, `PSA_MIL`, or `SC_MIL` only after checking coordinate
support and memory risk.

## Metrics

Classification:

- primary: macro AUC for multiclass or AUC for binary, unless user specifies otherwise
- secondary: balanced accuracy, macro F1, accuracy, confusion matrix
- report per-class performance when classes are imbalanced

After CV or seed sweeps finish in this repo, run:

```powershell
python -m auto_mil.cli analyze-stats --checkpoint runs\my_experiment\case_level_cv\checkpoint.json --metric test_macro_auc --baseline AB_MIL
```

Use the resulting mean/std/95% CI and paired comparisons for manuscript tables.

Survival:

- primary: C-index
- secondary: integrated Brier score, time-dependent AUC, calibration when available

Regression:

- primary: MAE or RMSE
- secondary: R2, Spearman/Pearson, residual plots

## Confirmation Summary Template

Before running baselines, present:

- task endpoint and prediction unit
- data counts and label distribution
- proposed split with rationale
- baseline suite
- metrics
- runtime budget
- expected artifacts
