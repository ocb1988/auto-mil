---
name: pathology-mil-experiment
description: Plan, run, iterate, and write manuscript-grade experimental sections for pathology AI projects using MIL, WSI features, MIL_BASELINE, or this auto-mil repository. Use when the user wants help with pathology classification, prognosis/survival, regression, baseline selection, split design, autonomous model innovation, hyperparameter tuning, ablation studies, experiment logging, git-traceable research workflows, or drafting Methods/Experiments/Results text for a pathology AI paper.
---

# Pathology MIL Experiment

## Core Rule

Run this as a staged research workflow with explicit user confirmations at decision gates. Do not jump straight into training unless the task, data, environment, split plan, baseline plan, and innovation plan are already clear and confirmed.

## Required Workflow

1. Clarify the task.
   - Ask for data location, label file, feature format, outcome type, and target endpoint.
   - Classify the problem as classification, prognosis/survival, regression, or another endpoint.
   - Confirm whether the goal is a quick demo, a conference-quality experiment, or manuscript-grade evidence.

2. Confirm the runtime.
   - Ask for local vs SSH/remote execution.
   - Confirm conda/pip path, Python executable, CUDA version, GPU availability, torch version, and intended working directory.
   - Verify with shell commands when possible. Record exact commands and outputs.

3. Explore the data before designing experiments.
   - Inspect slide/case counts, labels, missingness, class balance, center/cohort fields, patient/slide multiplicity, feature dimensions, coordinate availability, and leakage risks.
   - For H5/PT WSI feature bags, inspect keys and shapes from representative files.
   - Save exploration artifacts under a run directory.

4. Propose the split design and stop for confirmation.
   - If multiple datasets or centers exist, prefer center-aware design: train/validation on one or more centers and external test on held-out center(s).
   - If only one dataset exists, prefer patient-level n-fold cross validation; use a separate test split only when justified.
   - Always split by patient/case, not by slide, unless the task is truly slide-level and the user accepts the leakage risk.
   - For survival/regression, stratify or balance by event/risk/outcome quantiles when feasible.

5. Propose baselines and stop for confirmation.
   - For manuscript-grade pathology MIL, do not rely only on weak pooling baselines. Build a baseline suite that includes classic attention MIL, transformer/long-context MIL, and recent competitive MIL methods supported by the local repo.
   - Default classification candidates: sanity/light methods `MEAN_MIL` and `MAX_MIL`; classic MIL `AB_MIL` and optionally `GATE_AB_MIL` or `CLAM_SB_MIL`; strong classic/context methods `TRANS_MIL` and `RRT_MIL`; recent methods such as `STABLE_MIL` and `GDF_MIL` when their configs, dependencies, coordinates, and runtime requirements are satisfied.
   - If runtime is tight, run an initial 4-5 method screen such as `AB_MIL`, `TRANS_MIL`, `RRT_MIL`, `STABLE_MIL`, and `GDF_MIL`, with `MEAN_MIL`/`MAX_MIL` as cheap sanity checks when feasible. For final manuscript comparison, prefer a broader table that contains both classic and recent methods.
   - Verify each candidate against the available `MIL_BASELINE` configs/modules before promising it. Note special inputs such as coordinates for spatial methods.
   - State budget: epochs, folds, seeds, metrics, early stopping, and expected runtime.

6. Run baseline screen.
   - Use `auto_mil.cli` when the repo already supports the data/task.
   - Otherwise adapt the repo minimally, keeping generated configs, stdout, metrics, and checkpoints in a run directory.
   - Commit or at least record git status before and after material code changes.

7. Propose 1-2 innovation points and stop for confirmation.
   - Read the relevant MIL_BASELINE modules/configs before proposing changes.
   - Innovations may be module-level, training-method-level, loss-level, data-sampling-level, or aggregation-level.
   - Keep the first innovation small enough to test and ablate cleanly.

8. Autonomous improvement loop.
   - After the user confirms the innovation plan, ask for or define an autonomous run window: wall-clock time limit, maximum number of runs, target metric, allowed search space, hardware constraints, and stopping criteria.
   - Within that confirmed window, run continuously without asking for confirmation after every round, as long as changes stay inside the approved innovation/search scope and do not change the split, data, primary metric, or manuscript claims.
   - For each round: propose idea -> implement or configure -> run -> parse metrics -> reflect -> keep/revert/refine.
   - Optimize against the confirmed primary metric, but watch secondary metrics and overfitting.
   - For failed runs, classify the stdout/stderr with the repo log analyzer, then consult the failure policy before choosing the next action. Retry only when the policy keeps the approved data, split, metric, method family, and compute budget intact.
   - Keep a structured journal entry for every run, including command, config, code diff/commit, metric table, interpretation, and next action.
   - Continue until the target is reached, budget is exhausted, improvement plateaus, a safety/data-leakage risk appears, or the user stops the run.

9. Ablation study.
   - After finalizing the innovation, run ablations that isolate each component.
   - Include baseline, full method, and one row per removed/changed component.
   - Use the same split, seed policy, metrics, and reporting style as the main comparison.

10. Manuscript write-up.
   - Draft Methods and Experiments/Results in manuscript style after the evidence is stable.
   - Report dataset, preprocessing, split, baselines, implementation details, metrics, statistical treatment, main results, ablations, and limitations.
   - Do not overclaim from a single split or a small pilot.

## References

Load these only when needed:

- `references/experiment-design.md`: split design, baseline selection, metrics, and confirmation gates.
- `references/innovation-loop.md`: autonomous model-idea iteration, reflection, ablation design, and git hygiene.
- `references/manuscript-writing.md`: manuscript-style Methods and Results structure.

## Script

Use `scripts/init_experiment_record.py` to create a run record skeleton before a new manuscript-grade experiment:

```powershell
python skills\pathology-mil-experiment\scripts\init_experiment_record.py --run-dir runs\my_experiment --title "My pathology MIL experiment"
```

The script creates `experiment_plan.md`, `experiment_journal.jsonl`, and `git_snapshots/`.

## Confirmation Gates

Always pause for user confirmation after:

- task/data/runtime exploration summary
- split design
- baseline suite and budget
- proposed innovation(s)
- autonomous loop budget, target metric, search scope, and stopping criteria
- ablation matrix
- final manuscript claims

If the user explicitly approves a time-boxed autonomous loop, do not stop at every round for confirmation. Still log each decision and pause if the run needs to change data, split, primary metric, method family, compute budget, or final claims.
