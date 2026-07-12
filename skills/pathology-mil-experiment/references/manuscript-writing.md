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

