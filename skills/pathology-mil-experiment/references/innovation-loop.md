# Innovation Loop Reference

## Innovation Constraints

Propose 1-2 innovations at a time. Each innovation must be:

- testable against the confirmed baseline
- ablatable
- compatible with the available data and runtime
- not just a hyperparameter rename

Good innovation categories:

- module: attention gating, multi-scale aggregation, prototype memory, spatial graph, uncertainty weighting
- training: balanced sampling, curriculum, pseudo-bag strategy, hard instance mining, augmentation/mixup
- loss: class-balanced loss, focal loss, supervised contrastive auxiliary loss, calibration loss
- aggregation: patient-level aggregation across multiple slides
- robustness: center/stain-invariant regularization when centers exist

## Loop Protocol

Before entering the loop, confirm an autonomous run contract with the user:

- wall-clock budget, such as 2 hours, overnight, or until a specified date/time
- maximum number of experiments or training jobs
- target metric and minimum meaningful improvement threshold
- approved search space: model components, losses, samplers, hyperparameters, and seeds/folds
- fixed constraints: data, split, primary metric, baseline table, GPU/CPU resources, checkpoint policy
- stopping criteria: target reached, no improvement after N rounds, repeated failure, instability, leakage risk, or user interruption

After this contract is confirmed, continue running rounds without asking for confirmation after every idea, as long as every change stays inside the approved search space. Pause and ask again before changing the task definition, data, split, primary metric, method family, compute budget, or manuscript claims.

For each round:

1. State hypothesis.
2. Identify code/config changes.
3. Record git status or commit hash before changes.
4. Implement minimal change.
5. Run experiment.
6. Parse metrics into a table.
7. Reflect:
   - improved primary metric?
   - changed secondary metrics?
   - overfit signal?
   - unstable or data-leaky?
   - if failed, what log diagnosis category was assigned?
8. Decide:
   - keep and extend
   - tune locally
   - revert or abandon

During an autonomous window:

- record each hypothesis, command, result, reflection, and next action before starting the next run
- prefer small, interpretable changes over broad random search
- periodically summarize the best result so far in the experiment journal
- keep failed runs; mark them as failed with the error and corrective action
- run or read log diagnosis for each failed run, then apply the repo failure policy before choosing the next experiment
- allow automatic retry only for policy-approved categories such as CUDA OOM, NaN loss, timeout, or interruption, and only within the confirmed autonomous window
- pause for user confirmation before dependency installs, data path fixes, H5-key changes, shape-contract changes, split-policy changes, or ambiguous failures
- stop early if the target is reached with stable validation/test behavior, not just a noisy single-fold gain

When using this repo after the contract is confirmed:

```powershell
python -m auto_mil.cli run-autonomous-window --config configs\my_experiment.yaml --max-minutes 120 --max-runs 20 --target-metric test_macro_auc --target-value 0.85 --split-plan runs\my_experiment\split_plan\split_plan.json --plan-id patient_stratified_holdout --resume
```

Use `--timeout-seconds` for risky methods. Keep `--split-plan` and `--plan-id`
fixed throughout the autonomous window.

To preview or append new experiment-tree ideas from current evidence:

```powershell
python -m auto_mil.cli propose-nodes --config configs\my_experiment.yaml --max-proposals 6 --no-apply
```

Remove `--no-apply` only after the generated `proposal_report.md` stays within
the confirmed search scope.

## Git Hygiene

Before material code changes:

```powershell
git status --short
git diff --stat
```

After a coherent change and successful smoke test:

```powershell
git status --short
git diff --stat
```

If the user wants commits, make small commits:

- `exp: add prototype gated mil module`
- `exp: run cptac brca baseline sweep`
- `exp: add ablation runner`

Do not revert user changes unless explicitly asked.

## Experiment Journal Fields

Each run should record:

- run id
- timestamp
- git commit or diff summary
- task/split
- model/config path
- command
- environment
- stdout/stderr paths
- metric CSV path
- primary metric
- key secondary metrics
- interpretation
- next action

## Camyla-Style Proposal Discipline

Before entering the innovation loop, run a pre-innovation literature/proposal
step. Ask the user for seed papers, method drafts, or proposal notes first. If
the user provides them, treat those sources as the evidence base. If not, run
one lightweight online search and save the resulting proposal seeds:

```powershell
python -m auto_mil.cli literature-search --config configs\my_experiment.yaml
```

With user-provided sources:

```powershell
python -m auto_mil.cli literature-search --config configs\my_experiment.yaml --user-sources papers.json --offline
```

Do not perform online literature search at every experimental round. Refresh the
search only when the method track plateaus, the proposal is judged infeasible,
or the user explicitly asks for new literature. Keep the search report as
evidence and ask the user to confirm the selected proposal(s) before code
changes.

For method-track ideas, write a proposal record before coding:

- `name`: concise method name
- `track`: `method`
- `motivation`: pathology/MIL problem addressed
- `core_modules`: 1-3 named components with mechanisms
- `implementation_constraints`: feature shape, coordinate, memory, and runtime constraints
- `allowed_adaptations`: compatibility-only changes such as dimension, dropout, or learning-rate adjustment
- `forbidden_simplifications`: components that must not be replaced by identity/plain MLP/helper-only code
- `ablation_plan`: one component removed or disabled per row

Support-track changes can be run inside the same autonomous window, but label them
as `support` and keep them out of the main method claim. If a support change
beats the target, keep searching the method track unless the user changes the
goal.

When using this repo's experiment tree, optional method proposals can be placed
in config YAML:

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
      support_tags:
        - compatibility hyperparameters fixed before final comparison
      config_overrides: {}
```

The generated proposal report and experiment-tree report should show
`innovation_track`, `core_modules`, `support_tags`, and `ablation_plan`.

## Ablation Matrix

For a method with two innovations A and B:

| Row | A | B | Purpose |
|---|---|---|---|
| Baseline | no | no | reference |
| +A | yes | no | isolate A |
| +B | no | yes | isolate B |
| Full | yes | yes | final method |

Keep training budget and split identical. If runtime is tight, use the best fold or a small pilot only after stating the limitation.

When using this repo for the current AB_MIL innovation path:

```powershell
python -m auto_mil.cli run-ablation-cv --config configs\my_experiment.yaml --split-plan runs\my_experiment\split_plan\split_plan.json
```

The default matrix runs cross-entropy baseline, focal loss only, prototype head
only, and focal-plus-prototype full method on the confirmed CV split.
