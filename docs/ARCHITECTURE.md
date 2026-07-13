# Auto-MIL Architecture

This project adapts the autonomous research pattern in Camyla to pathology MIL.
It intentionally keeps the first implementation lightweight: the execution
backend is MIL_BASELINE, and the research policy is deterministic and auditable.
An LLM proposal generator can be added later without changing the training
runner interface.

## Camyla Concepts Reused

| Camyla idea | Auto-MIL equivalent |
|---|---|
| Dataset-in research entry | `prepare_cptac_brca` scans H5 bags and clinical labels |
| Task/data contract | `TaskSpec` and `DatasetSpec` normalize endpoint and feature inputs |
| Split confirmation gate | `split_plan.json/md` proposes center/external/CV plans |
| Baseline stage gate | `baseline_screen` runs cheap MIL_BASELINE models |
| QWBE-style candidate queue | `experiment_tree.json` with scored parent/child recipe nodes |
| Stage journal | `research_journal.jsonl` append-only records |
| Checkpoint/resume | `checkpoint.json` and `checkpoint_events.jsonl` per run directory |
| Log diagnosis | `LogDiagnosis` records attached to failed run payloads |
| Failure policy | `FailureAction` retry/escalation decisions from diagnoses |
| Experiment report | `report.md` with commands, metrics, artifacts |
| Reproducible workspace | generated YAMLs and stdout logs per recipe |

## Current Workflow

1. `spec_inspection`
   - `inspect-spec` prints the normalized task/data interface
   - verifies whether the current MIL_BASELINE adapter can prepare the task
   - records feature format, feature key, coordinate key, case id policy, and
     label/target fields

2. `dataset_audit`
   - normalizes config into `TaskSpec` and `DatasetSpec`
   - scans H5 files
   - infers feature dimension from the configured feature key
   - joins slide files to `case_id`
   - splits by case to avoid patient leakage
   - writes MIL_BASELINE wide CSV

3. `split_planning`
   - `plan-split` inspects the matched case table before baseline execution
   - prioritizes predefined external test, center-aware holdout, or n-fold CV
   - writes `split_plan.json` and `split_plan.md` for user confirmation

4. `baseline_screen`
   - runs low-budget models such as `MEAN_MIL`, `MAX_MIL`, `AB_MIL`
   - stores one generated YAML per recipe
   - parses `Best_Log*.csv`

5. `focused_runs`
   - selects the best screened model as the anchor
   - expands learning-rate/dropout/balanced-sampler recipes
   - runs a longer budget

6. `report`
   - ranks all completed runs by `test_macro_auc`, falling back to
     `val_macro_auc`
   - records exact commands and artifact paths

7. `experiment_tree`
   - seeds root nodes from candidate baseline models
   - executes pending nodes selected by a QWBE-lite score
   - expands high-scoring root nodes into focused hyperparameter children
   - creates conservative retry children for selected failure categories
   - writes `experiment_tree.json` and `experiment_tree.md`

## Checkpoint and Resume

Each long-running command writes a checkpoint under its output directory:

- `run`: `runs/<experiment>/checkpoint.json`
- `run-cv`: `runs/<experiment>/case_level_cv/checkpoint.json`
- `run-innovation-cv`: `runs/<experiment>/innovation_cv/checkpoint.json`
- `run-tree`: `runs/<experiment>/experiment_tree/checkpoint.json`

The checkpoint stores one record per recipe/run id, including status, command,
artifact paths, metrics, and error text. `--resume` reuses only records marked
`completed`; failed and dry-run records are treated as needing another run. This
keeps interrupted overnight searches recoverable without silently trusting
failed or incomplete experiments.

## QWBE-Lite Tree Search

`run-tree` is the first search layer toward Camyla-style experiment
orchestration. It keeps each candidate as an `ExperimentNode` with:

- parent id, child ids, depth, prior, status, visits, and score
- a concrete `Recipe` that can be executed by MIL_BASELINE
- a rationale string explaining why the node exists

Root nodes screen baseline/model families. Completed high-scoring roots are
expanded into focused child nodes over local hyperparameters such as learning
rate, dropout, and balanced sampling. The selector uses a compact PUCT-style
score so future LLM-generated priors can plug in without changing the execution
contract.

## Log Diagnosis

Failed runs are passed through a lightweight failure classifier. The diagnosis
is written into the run payload and, when available, as a sibling
`*.diagnosis.json` next to the stdout log. Current classes include:

- `missing_dependency`
- `cuda_oom`
- `cuda_runtime`
- `metric_missing_class`
- `file_not_found`
- `h5_key_error`
- `shape_mismatch`
- `nan_loss`
- `timeout`
- `keyboard_interrupt`
- `python_exception`
- `unknown`

The classifier is intentionally rule-based for now. It creates a stable
interface for later autonomous decisions such as install dependency, reduce
memory pressure, rerun with safer split, or escalate to the user.

## Failure Policy

`failure_policy.py` consumes a `LogDiagnosis` and returns a `FailureAction`.
Retryable categories are limited to failures where the experiment definition can
remain fixed:

- `cuda_oom`: retry with lower memory-pressure config overrides
- `nan_loss`: retry with lower learning rate and conservative regularization
- `timeout`: retry with a shorter epoch budget
- `keyboard_interrupt`: resume or rerun through checkpoint policy

The policy pauses for dependency installs, data path/H5-key errors, shape
mismatches, missing-class metrics, and ambiguous Python/CUDA failures. This is
intentional: autonomous loops should not silently change data, split policy,
runtime packages, primary metrics, or manuscript claims.

## Why This Shape

MIL_BASELINE already provides many model implementations and a common training
surface. Auto-MIL should therefore be a research orchestrator rather than a new
training framework. The code treats MIL_BASELINE as the executable substrate and
keeps the autonomous research state outside it.

## Task/Data Interface

`specs.py` defines the front door for new projects:

- `TaskSpec` captures classification, prognosis/survival, and regression
  fields, plus split seed and default split ratios.
- `DatasetSpec` captures dataset paths, case id columns, center/cohort/external
  test columns, H5 feature key, coordinate key, and case-id filename regex.
- `FeatureSpec` records the feature-bag format and key names.

The current executable path supports classification with H5 feature bags. Other
task kinds are represented but intentionally blocked at preparation time until a
matching runner/metric adapter exists.

## Split Planner

`split_planner.py` converts the normalized task/data interface and matched case
table into a confirmation artifact:

- predefined external-test columns become an external-test proposal
- center columns become center-holdout proposals
- single-cohort datasets default to patient-level stratified n-fold CV
- train/val/test holdout remains available for pilots and demos

The planner is deliberately separate from execution. It should be reviewed and
confirmed before baseline runs so later autonomous loops cannot silently change
the data split after seeing results.

## Next Extensions

- Add an LLM proposal stage that writes `ExperimentNode` objects from literature/context.
- Add split execution adapters that consume a confirmed split plan directly.
- Add spatial recipe families for `DAG_MIL`, `PSA_MIL`, `SC_MIL`, and
  coordinate-aware models.
- Add seed sweeps and statistical comparison across top recipes.
- Add case-level prediction aggregation for datasets with multiple slides per
  patient.
- Add paper-style report generation once experiment evidence is substantial.
