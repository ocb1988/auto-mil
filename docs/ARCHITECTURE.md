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
| Baseline stage gate | `baseline_screen` runs cheap MIL_BASELINE models |
| QWBE-style candidate queue | `experiment_tree.json` with scored parent/child recipe nodes |
| Stage journal | `research_journal.jsonl` append-only records |
| Checkpoint/resume | `checkpoint.json` and `checkpoint_events.jsonl` per run directory |
| Log diagnosis | `LogDiagnosis` records attached to failed run payloads |
| Experiment report | `report.md` with commands, metrics, artifacts |
| Reproducible workspace | generated YAMLs and stdout logs per recipe |

## Current Workflow

1. `dataset_audit`
   - scans H5 files
   - infers feature dimension from `features`
   - joins slide files to `case_id`
   - splits by case to avoid patient leakage
   - writes MIL_BASELINE wide CSV

2. `baseline_screen`
   - runs low-budget models such as `MEAN_MIL`, `MAX_MIL`, `AB_MIL`
   - stores one generated YAML per recipe
   - parses `Best_Log*.csv`

3. `focused_runs`
   - selects the best screened model as the anchor
   - expands learning-rate/dropout/balanced-sampler recipes
   - runs a longer budget

4. `report`
   - ranks all completed runs by `test_macro_auc`, falling back to
     `val_macro_auc`
   - records exact commands and artifact paths

5. `experiment_tree`
   - seeds root nodes from candidate baseline models
   - executes pending nodes selected by a QWBE-lite score
   - expands high-scoring root nodes into focused hyperparameter children
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

## Why This Shape

MIL_BASELINE already provides many model implementations and a common training
surface. Auto-MIL should therefore be a research orchestrator rather than a new
training framework. The code treats MIL_BASELINE as the executable substrate and
keeps the autonomous research state outside it.

## Next Extensions

- Add an LLM proposal stage that writes `ExperimentNode` objects from literature/context.
- Add policy actions that consume `LogDiagnosis` and automatically retry or adapt failed nodes.
- Add spatial recipe families for `DAG_MIL`, `PSA_MIL`, `SC_MIL`, and
  coordinate-aware models.
- Add seed sweeps and statistical comparison across top recipes.
- Add case-level prediction aggregation for datasets with multiple slides per
  patient.
- Add paper-style report generation once experiment evidence is substantial.
