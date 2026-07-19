# Vendored Outcome-Task Adaptation Audit

## Scope And Rule

Audit target: the 50 model directories under `third_party/MIL_BASELINE/modules`
as shipped in this repository.

Every model has a matching config and process directory. All current vendored
`process/*` runners are classification-only: they construct `num_classes`
logits, use the classification data/metric loops, and select checkpoints with
classification metrics. That fact does not mean the underlying MIL aggregator
is classification-only.

This audit separates the vendored **module** from its current classification
**trainer**. A `Head/loss` status means the module exposes `WSI_feature` and its
bag aggregation can be retained while Auto-MIL supplies a one-output regression
or risk head plus MSE/Cox loss. It does not mean that the existing vendored
process can be called unchanged.

## Status Legend

- `Head/loss`: direct outcome candidate. Keep the patch encoder and aggregator,
  replace the classifier with a scalar task head, and use the outcome trainer.
- `Head/loss + input`: same adaptation, but coordinates or a required external
  asset must be passed through the outcome data adapter. Pseudo-grid fallback is
  not equivalent to real coordinates.
- `Structural`: its aggregation path selects class-specific branches or needs
  multiple classification heads. It needs a dedicated task-aware redesign, not
  a simple head replacement.
- `Training-policy`: its module is close to AB-MIL, but the method contribution
  is a classification mixup/pseudo-label policy. Regression and survival need
  separate target-mixing or risk-set semantics before it is a valid baseline.

## Per-Method Results

| Method | Status | Outcome-task assessment |
|---|---|---|
| `AB_MIL` | Head/loss | Attention bag feature is exposed as `WSI_feature`; supported now. |
| `AC_MIL` | Head/loss | Multi-token attention produces `feat_bag`; ignore class-specific auxiliary predictions and attach a scalar head. |
| `ADD_MIL` | Head/loss | Attention aggregation is reusable; instance classifier is not required for an outcome head. |
| `AEM_MIL` | Head/loss | Entropy regularization is task-agnostic; replace final classifier and retain it only when explicitly configured. |
| `AMD_MIL` | Head/loss | Agent-based bag representation is exposed; replace `_fc2`. |
| `CA_MIL` | Head/loss | Context-aware bag feature is reusable; class logits are terminal only. |
| `CDP_MIL` | Head/loss | Dirichlet-process representation can feed a scalar outcome head; audit its variational regularizer when enabling it. |
| `CLAM_SB_MIL` | Structural | Optional instance supervision is class-label/one-hot based. Disable it only for a stripped backbone, or redesign the instance objective. |
| `CLAM_MB_MIL` | Structural | Multi-branch attention and per-class bag/instance classifiers are class-conditional; needs a task-specific aggregation redesign. |
| `DAG_MIL` | Head/loss + input | Dynamic adjacency aggregation is reusable; use real coordinates for a valid spatial experiment. |
| `DG_MIL` | Head/loss | Distribution-guided representation can feed an outcome head. |
| `DGR_MIL` | Head/loss | Lesion/region aggregation is reusable; preserve the chosen `bag_mode` in the outcome config. |
| `DS_MIL` | Structural | Bag feature and attention are selected using the predicted class (`Y_hat`); replace class-conditioned critical-instance selection with a scalar/task-neutral rule. |
| `DT_MIL` | Head/loss | Deformable Transformer bag feature is exposed; replace final classifier. |
| `DTFD_MIL` | Structural | The runner composes four modules and uses class-score-based instance selection/distillation with two optimizers. Requires a dedicated outcome trainer. |
| `DYHG_MIL` | Head/loss | Dynamic hypergraph bag representation is reusable. |
| `FOURIER_MIL` | Head/loss | Fourier aggregation is task-agnostic before the final classifier. |
| `FR_MIL` | Head/loss | Feature-recalibration representation is exposed; replace final classifier. |
| `GATE_AB_MIL` | Head/loss | Gated attention bag feature is reusable; supported through Auto-MIL's task-aware implementation. |
| `GDF_MIL` | Head/loss | Graph/decomposition feature is exposed; supported now. Its optional smooth/NCE terms require separate validation before use. |
| `IB_MIL` | Head/loss | Information-bottleneck aggregation is reusable; keep beta regularization only after outcome-task validation. |
| `IIB_MIL` | Head/loss | Query/decoder bag feature is exposed; replace WSI classifier. |
| `ILRA_MIL` | Head/loss | Latent representation aggregator can use a scalar head. |
| `INSMIX_MIL` | Training-policy | Instance-level mixup is implemented in the classification process. Regression may mix continuous targets; Cox survival should not use it without a justified survival formulation. |
| `LONG_MIL` | Head/loss + input | Long-context aggregator is reusable, but needs the configured ALiBi position-embedding asset. |
| `MAMBA_MIL` | Head/loss | State-space bag feature is exposed; replace final classifier. |
| `MAMBA2D_MIL` | Head/loss + input | Use patch coordinates or a documented grid construction; then replace the classifier. |
| `MAX_MIL` | Head/loss | Max-pooled bag feature is reusable; sanity baseline. |
| `MEAN_MIL` | Head/loss | Mean-pooled bag feature is reusable; sanity baseline. |
| `MHIM_MIL` | Head/loss | Instance masking is label-free; retain it as an augmentation policy and replace predictor. |
| `MICO_MIL` | Head/loss | Cluster-reduction representation is usable; its cluster/self-supervised terms need independent outcome validation. |
| `MICRO_MIL` | Head/loss | Micro-cluster representation is exposed; replace classifier. |
| `MIXUP_MIL` | Training-policy | Standard bag mixup uses classification soft targets in its process. It is straightforward for regression targets but needs a new survival objective. |
| `MO_MIL` | Head/loss + input | Multi-order aggregation is reusable; coordinate/sequence options must be retained in the adapter. |
| `MSM_MIL` | Head/loss | Mamba-style sequence aggregation is reusable. |
| `NCIE_MIL` | Head/loss | Channel/spatial interaction representation is reusable; replace its `n_classes` head. |
| `PA_MIL` | Head/loss | Pyramid/attention bag feature is reusable. |
| `PGCN_MIL` | Head/loss | Graph-convolution bag representation is reusable; record whether the DGL or fallback path is used. |
| `PSA_MIL` | Head/loss + input | Spatial-prior attention needs coordinates for a scientifically valid result; pseudo-grid is only a fallback. |
| `PSEBMIX_MIL` | Training-policy | Pseudo-bag mixing and its target semantics are classification-specific in the current process; redesign separately for regression/survival. |
| `RANKMIX_MIL` | Training-policy | Rank-based mixup relies on classification labels/ranking policy; define outcome-specific ranking before use. |
| `REMIX_MIL` | Training-policy | Feature-level mixing process is tied to classification loss/soft labels; needs task-specific loss semantics. |
| `RET_MIL` | Head/loss | Retention-based bag feature is reusable; replace classifier. |
| `RRT_MIL` | Head/loss | Re-embedding plus attention aggregation is reusable; supported now. |
| `S4_MIL` | Head/loss | S4 sequence feature is reusable; replace classifier. |
| `SC_MIL` | Head/loss + input | Sparse context aggregation should receive true patch coordinates for clustering. |
| `STABLE_MIL` | Head/loss + input | Spatial-stable aggregation is reusable; supported now. Real coordinates are preferred over its pseudo-grid fallback. |
| `TDA_MIL` | Head/loss | Topological/deformable attention feature is reusable; replace classifier. |
| `TRANS_MIL` | Head/loss | CLS bag feature is reusable; supported now. |
| `WIKG_MIL` | Head/loss | Knowledge-guided graph bag feature is reusable; replace final classifier. |

## Counts

- `41` head/loss candidates, including `7` that require coordinates or an
  external positional asset.
- `4` structural adaptations: `CLAM_SB_MIL`, `CLAM_MB_MIL`, `DS_MIL`, and
  `DTFD_MIL`.
- `5` classification training-policy adaptations: `INSMIX_MIL`, `MIXUP_MIL`,
  `PSEBMIX_MIL`, `RANKMIX_MIL`, and `REMIX_MIL`.

Thus no method should be described as inherently impossible for an outcome
task. However, all 50 are classification-only **as their vendored processes are
currently written**, and the nine structural/training-policy methods cannot be
reported as direct outcome baselines without the listed dedicated adaptation.

Auto-MIL now implements the 41 `Head/loss` candidates through a generic
vendored-backbone adapter. It reads the matching model config, passes through
coordinates where available, requests `WSI_feature`, and attaches a scalar
outcome head. `CDP_MIL`, `NCIE_MIL`, and `RET_MIL` have explicit adapter handling
for their non-standard feature, fixed-grid, and hidden-feature interfaces.
Current-environment smoke testing completed forward/backward passes for 35 of
the 41. The remaining six require external runtime components: `DT_MIL`
(compiled MSDeformAttn), `LONG_MIL` (`xformers` and its position asset),
`MAMBA_MIL`, `MAMBA2D_MIL`, and `MSM_MIL` (`mamba-ssm`), and `MICRO_MIL`
(`dgl`). `PGCN_MIL` and `WIKG_MIL` are now verified through
`torch-geometric`.

## Auto-MIL Implementation Priority

1. Use the five validated models `AB_MIL`, `TRANS_MIL`, `RRT_MIL`,
   `STABLE_MIL`, and `GDF_MIL` only as a functional starter screen, plus pooling
   sanity baselines. For a manuscript comparison, select exactly five task-valid
   methods through a documented literature review that prioritizes the preceding
   1-2 years and high-impact journals/conferences; do not treat this starter list
   as the final comparison by default.
2. Add per-method outcome smoke tests and install the remaining documented optional
   dependencies only when a project selects those methods.
3. Keep real patch coordinates enabled before reporting coordinate-aware methods
   as manuscript baselines.
4. Design and validate `CLAM`, `DS_MIL`, and `DTFD_MIL` as separate methods;
   do not silently disable their defining classification components.
5. Treat mixup methods as training-method research, not interchangeable model
   names. Implement regression/survival variants only with an explicit
   methodological rationale and ablation.
