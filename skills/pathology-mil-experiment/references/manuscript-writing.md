# Manuscript Writing Reference

## Methods: Experiments

Include:

- datasets/cohorts, inclusion criteria, number of patients/slides, centers
- labels/outcomes and prediction unit
- feature extraction source and feature dimensions
- split design and leakage prevention
- baseline models and why selected
- proposed method summary
- implementation details: optimizer, learning rate, epochs, batch size, hardware, seeds
- evaluation metrics and statistical protocol

## Results

Order:

1. Dataset/exploration summary.
2. Main comparison table.
3. External validation or cross-validation summary.
4. Ablation table.
5. Qualitative analysis if available, such as attention heatmaps.
6. Error analysis and limitations.

When using this repo, prefer generated evidence artifacts:

- `manuscript_results.md` and `model_summary.csv` for main result tables
- `stats_report.md` for confidence intervals and paired comparisons
- `prediction_report.md` and `case_predictions.csv` for case-level predictions
- `figure_report.md`, `per_class_metrics.csv`, and `error_cases.csv` for
  ROC/confusion matrix/calibration/error analysis claims

Generate the first evidence-indexed draft with:

```powershell
python -m auto_mil.cli write-manuscript --root runs\my_experiment --config configs\my_experiment.yaml
```

Then package the draft for polishing and submission checks:

```powershell
python -m auto_mil.cli package-manuscript --root runs\my_experiment --profile generic-pathology-ai --target-journal "target journal name"
```

Treat `manuscript/manuscript_draft.md` as a scaffold, not a final paper section.
Use its warnings and claim guardrails to decide which experiments, statistics,
or figures are still missing before polishing the prose. Treat
`manuscript/package/llm_polish_prompt.md` as an evidence-bounded prompt: do not
allow the polishing step to add claims not supported by the evidence manifest.

## Tone

Use precise claims:

- "improved macro AUC from X to Y on the held-out test set"
- "showed consistent gains across N folds"
- "pilot result; requires repeated-seed validation"

Avoid unsupported claims:

- "state-of-the-art" unless compared fairly to strong published baselines
- "clinically deployable" without external validation and calibration
- "robust" without center/stain/seed stress tests

## Table Template

| Method | Val AUC | Test AUC | BACC | Macro F1 |
|---|---:|---:|---:|---:|
| MEAN_MIL |  |  |  |  |
| MAX_MIL |  |  |  |  |
| AB_MIL |  |  |  |  |
| Ours |  |  |  |  |

## Ablation Text Template

"To assess the contribution of each component, we removed [component A] and [component B] from the full model while keeping the same data split, training budget, and evaluation protocol. Removing [component] reduced [metric] from [full] to [ablated], suggesting that [interpretation]."
