---
name: pathology-mil-experiment
description: Plan, run, iterate, package, and write manuscript-grade experimental sections for pathology AI projects using MIL, WSI features, MIL_BASELINE, or this auto-mil repository. Use when the user wants help with pathology classification, prognosis/survival, regression, baseline selection, split design, autonomous model innovation, hyperparameter tuning, ablation studies, experiment logging, git/GitHub-traceable research workflows, manuscript evidence packaging, or drafting Methods/Experiments/Results text for a pathology AI paper.
---

# Pathology MIL Experiment

## Core Rule

Run this as a staged research workflow with explicit user confirmations at decision gates. Do not jump straight into training unless the task, data, environment, split plan, baseline plan, and innovation plan are already clear and confirmed.

## Reusable Core and Run Workspace Rule

Treat `auto_mil/`, `third_party/MIL_BASELINE/`, and shared root-level configs as a reusable, versioned core. Do not edit them for the needs of one dataset or one paper experiment. Before data exploration, create one project workspace at `runs/<project_slug>/project/` and keep all project-specific state there. Keep transient training outputs under the same `runs/<project_slug>/` root, outside `project/`.

Use this layout:

```text
runs/<project_slug>/
  project/                    # durable, git-trackable project definition
    project_manifest.json     # task, data references, environment, core git revision
    experiment_plan.md
    experiment_journal.jsonl
    configs/                  # immutable per-project/per-run config copies
    src/                      # project-only adapters, models, losses, and utilities
    patches/                  # documented patches/overrides when an adapter is impossible
    records/                  # split, proposal, and manuscript evidence records
  artifacts/                  # ignored logs, checkpoints, predictions, figures, reports
```

Put every dataset- or paper-specific change under that workspace, including label normalization, ID mapping, feature readers, data adapters, custom model/loss modules, per-project command wrappers, configs, split files, proposals, and manuscript artifacts. Never modify a vendored baseline in place for a project experiment; wrap or override it from `runs/<project_slug>/project/src/` and record the exact invocation.

Keep source/config/record files in the run workspace eligible for git tracking; keep large features, checkpoints, caches, logs, and generated predictions ignored. Each run must record the core commit, dirty-state summary, project source/config paths, and commands so that it can be replayed against the same core revision.

Change the reusable core only when the change is demonstrably generic across projects. Make that a separate, tested, explicitly described core commit before or after the project experiment; do not hide a dataset-specific workaround inside `auto_mil/` or `third_party/MIL_BASELINE/`.

If the core lacks an extension point needed by a project, first propose a generic plugin/adapter interface and obtain approval for it as core work. Do not bypass this rule by inserting the project's model code into a shared training module.

## Innovation Integrity Rule

Prioritize manuscript-worthy method innovation over engineering or tuning gains. Treat network architecture changes, MIL aggregation mechanisms, representation learning modules, pathology-aware inductive biases, objective/loss design, or principled training strategies as candidate method innovations. Treat hyperparameter tuning, ensembling, test-time aggregation, threshold tuning, seed search, longer training, data caching, and runtime optimizations as engineering/supportive techniques unless they are part of a clearly stated scientific method.

Do not present engineering gains as the paper's main algorithmic contribution. Keep them in a separate "engineering/tuning/ensemble" track, useful for robustness, upper-bound analysis, or deployment, but not as the primary method claim. If the best metric comes from an ensemble or tuning-only recipe, report it honestly as auxiliary evidence and continue searching for a single-model or principled method innovation unless the user explicitly changes the goal.

Use Camyla-style proposal discipline for method innovation:

- Treat Stage 1 as baseline only: use standard/default settings and do not tune it into a hidden method.
- Treat Stage 2 as complete research-proposal testing: each method proposal must name 1-3 coherent core modules, explain the mechanism, state the expected contribution, and define what ablation would remove.
- Preserve the proposal's fundamental concept during implementation. You may adjust dimensions, learning rates, dropout, or other compatibility details, but do not replace a proposed core module with an identity/plain MLP/helper-only change and still call it the same method.
- Treat Stage 3 as ablation verification: remove or disable exactly one method component at a time while keeping the split, data, metric, and other components fixed.
- In writing, give Method subsections only to novel method components. Put optimizer, epoch, seed, threshold, ensemble, and hyperparameter details in implementation details or auxiliary analysis unless they are the declared scientific contribution.

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
   - When labels are patient/case-level and cases may have multiple slides, prefer constructing one case-level MIL bag by concatenating patch features from all slides before training; use slide-level output aggregation only as a secondary analysis or when the user explicitly wants slide-level MIL.
   - Save exploration artifacts under `runs/<project_slug>/artifacts/exploration/`; put any exploration-specific reader or ID-mapping code under `runs/<project_slug>/project/src/`.

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
   - Otherwise adapt through `runs/<project_slug>/project/src/` or `runs/<project_slug>/project/patches/`, keeping generated configs, stdout, metrics, and checkpoints under that project's run root. Do not make a dataset-specific edit to the reusable core.
   - Commit or at least record git status before and after material code changes. Record project-workspace source changes separately from any approved generic core change.

7. Propose 1-2 innovation points and stop for confirmation.
   - Before proposing method innovations, ask whether the user has seed papers, methods, or proposal drafts. Accept JSON/CSV lists, paper titles/URLs, copied abstracts, or a short written method idea.
   - If the user provides a local literature review or paper corpus, use it as the primary evidence base. Read its overview/manifest first, then only the closest individual method summaries needed to evaluate novelty. Record its path, scope, and evidence boundary in the project literature report; do not copy its PDFs into the Auto-MIL repo.
   - If the user provides sources, use them as the literature/proposal evidence. If not, run one pre-innovation literature/proposal search and save the report before drafting ideas. Use online search only to fill a documented coverage gap or refresh recent work.
   - In this repo, prefer:
     `python -m auto_mil.cli literature-search --config configs\my_experiment.yaml`
     Use `--user-sources papers.json` when the user provides a paper list. For an overview Markdown or corpus directory, use `--local-review <path> --offline`; `innovation.literature_search.local_review_path` can store the same path in a project config.
   - Read the relevant MIL_BASELINE modules/configs before proposing changes.
   - Build a novelty matrix for each proposed method: closest reviewed methods, shared mechanism, new mechanism, task/data-specific rationale, implementation risk, and one ablation per claimed component. Reject a proposal that is only a renamed reviewed method or an unprincipled stacking of reviewed modules.
   - Write each method innovation as a compact research proposal: motivation, core module(s), mechanism, expected contribution, implementation constraints, and ablation plan.
   - Classify each candidate as one of:
     - `method`: architecture, MIL aggregation, pathology-aware module, representation learning, loss/objective, or principled training mechanism suitable for a manuscript contribution.
     - `support`: hyperparameter tuning, sampling schedule, caching, efficiency, thresholding, ensemble, or other engineering/analysis aid.
   - Propose at least one `method` innovation before proposing support-only improvements.
   - Innovations may be module-level, training-method-level, loss-level, data-sampling-level, or aggregation-level, but explain the scientific rationale and what ablation would isolate.
   - Keep the first innovation small enough to test and ablate cleanly.
   - If using this repo's experiment tree, encode method proposals in `innovation.method_proposals` with `name`, `model_name`, `core_modules`, `ablation_plan`, optional `support_tags`, and minimal `config_overrides`, or let `propose-nodes` read `literature_search/literature_proposals.json` generated by the pre-search step.

8. Autonomous improvement loop.
   - After the user confirms the innovation plan, ask for or define an autonomous run window: wall-clock time limit, maximum number of runs, target metric, allowed search space, hardware constraints, and stopping criteria.
   - Separate the approved search scope into `method track` and `support track`. The method track is the primary route to the paper contribution; the support track can improve runtime, stability, or an auxiliary upper-bound result.
   - Within that confirmed window, run continuously without asking for confirmation after every round, as long as changes stay inside the approved innovation/search scope and do not change the split, data, primary metric, or manuscript claims.
   - For each round: propose idea -> implement or configure -> run -> parse metrics -> reflect -> keep/revert/refine.
   - In every reflection, state whether the gain came from a method change or a support/tuning/ensemble change.
   - Preserve core method modules across debugging. If a core module proves infeasible, refine the proposal and log the reason instead of silently simplifying it away.
   - If an ensemble, longer training, seed choice, or hyperparameter-only change reaches the target, mark the target as reached for the support track only. Continue method-track exploration within the remaining approved budget unless the user explicitly accepts support-track performance as the goal.
   - Optimize against the confirmed primary metric, but watch secondary metrics and overfitting.
   - For failed runs, classify the stdout/stderr with the repo log analyzer, then consult the failure policy before choosing the next action. Retry only when the policy keeps the approved data, split, metric, method family, and compute budget intact.
   - Keep a structured journal entry for every run, including command, config, code diff/commit, metric table, interpretation, and next action.
   - Continue until the target is reached, budget is exhausted, improvement plateaus, a safety/data-leakage risk appears, or the user stops the run.

9. Ablation study.
   - After finalizing the innovation, run ablations that isolate each method component.
   - Include baseline, full single-model method, and one row per removed/changed method component.
   - Put support-track additions such as ensembles, threshold tuning, longer training, or hyperparameter changes in a separate auxiliary table unless they are essential to the proposed method.
   - Use the same split, seed policy, metrics, and reporting style as the main comparison.

10. Manuscript write-up.
   - Draft Methods and Experiments/Results in manuscript style after the evidence is stable.
   - Report dataset, preprocessing, split, baselines, implementation details, metrics, statistical treatment, main results, ablations, and limitations.
   - Clearly separate the claimed method contribution from engineering/tuning/ensemble results. Do not call an ensemble, seed search, or hyperparameter recipe a new algorithm unless it includes a principled methodological mechanism and ablation support.
   - In this repo, prefer `write-manuscript` to create `manuscript_draft.md` and `manuscript_evidence.json`, then `package-manuscript` to create a polishing prompt and submission checklist.
   - Treat `llm_polish_prompt.md` as evidence-bounded: do not let polishing add claims not supported by the evidence manifest.
   - Do not overclaim from a single split or a small pilot.

11. Version-control handoff.
   - Keep project code changes, configs, records, and manuscript workflow changes under `runs/<project_slug>/project/` in small commits with descriptive messages. Do not commit large run artifacts.
   - Keep generic framework changes outside `runs/` in separate commits that identify why they generalize beyond the current project.
   - Before pushing, verify `git status`, `git remote -v`, and the intended GitHub account/repository with the user.
   - Never store GitHub passwords or personal access tokens in the repo or experiment logs.
   - Push only after the user confirms the target remote.

## References

Load these only when needed:

- `references/experiment-design.md`: split design, baseline selection, metrics, and confirmation gates.
- `references/innovation-loop.md`: autonomous model-idea iteration, reflection, ablation design, and git hygiene.
- `references/manuscript-writing.md`: manuscript-style Methods and Results structure.

## Script

Use `scripts/init_experiment_record.py` to create an isolated project workspace before a new manuscript-grade experiment:

```powershell
python skills\pathology-mil-experiment\scripts\init_experiment_record.py --run-dir runs\my_experiment --title "My pathology MIL experiment"
```

The script creates `project/` with the plan/journal/manifest plus `configs/`, `src/`, `patches/`, and `records/`; it also creates the ignored `artifacts/` directory. Treat `project/src/` as the only location for project-specific code.

## Confirmation Gates

Always pause for user confirmation after:

- task/data/runtime exploration summary
- split design
- baseline suite and budget
- proposed innovation(s)
- autonomous loop budget, target metric, search scope, and stopping criteria
- ablation matrix
- final manuscript claims
- remote repository or push target changes

If the user explicitly approves a time-boxed autonomous loop, do not stop at every round for confirmation. Still log each decision and pause if the run needs to change data, split, primary metric, method family, compute budget, or final claims.
