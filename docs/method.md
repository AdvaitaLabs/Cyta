# Method

## Problem and motivation

A gene perturbation changes a population of cells, not a single expression vector. A useful prediction must recover the affected genes, the direction and size of their responses, and the population-level composition. These requirements can disagree when optimized through one expression head.

For example, consider two cells with counts `[9, 1]` and `[10, 90]` for two genes. The first gene has a mean per-cell fraction of `(0.9 + 0.1) / 2 = 0.5`, but a pooled fraction of `19 / 110 ≈ 0.173`. A model can improve arithmetic mean fold changes while worsening pooled-response similarity. Advaita uses separate arithmetic-amplitude and pooled-response branches to address this difference.

The public recipe corresponds to the final Advaita V20 specialist revision. It retains numerical count generation and source-derived response evidence. It is a hybrid model, and the PIE architecture and external source representations are explicitly attributed.

## Foundation model

The backbone is PIE's multimodal Perceiver. Biological knowledge sources are encoded as source tokens; learned latent representations interact with gene queries, control features and cross-context response evidence. The outputs include DE logits, continuous log2 fold change, and discretized distribution-change logits.

The retained foundation configuration uses:

| Parameter | Value |
| --- | --- |
| Model dimension | 384 |
| Latent slots | 128 |
| Encoder / processor / decoder layers | 2 / 4 / 2 |
| Attention heads | 8 |
| Feed-forward multiplier | 2 |
| Model dropout / source dropout | 0.10 / 0.05 |
| Evidence encoder / output dimension | 64 / 128 |
| Evidence response dropout | 0.25 |
| Maximum source tokens during sampled training/inference | 128 per source |
| DE / distribution readout temperatures | 4.0 / 4.5 |

The complete configuration is in [`foundation_architecture.json`](../configs/foundation_architecture.json). The specialist's near-background batches retain their original full source-token sets; the 128-token cap applies to source minibatches and exported inference.

Foundation training starts from random weights. It first visits every eligible source record in context-homogeneous batches of 32, then performs 1,600 laboratory-balanced updates. The historical run used 159,367 eligible rows, 5,037 coverage steps and 6,637 total steps. Each update samples up to 768 genes uniformly, shared within the batch. AdamW uses learning rate `1e-4`, weight decay `0.01`, an 8% warmup, cosine decay to 5% of the initial rate and gradient clipping at 1.0. Tahoe receives sampling weight 0.20; the remaining laboratory groups share the remaining probability.

The objective adds dense expression supervision to PIE's original multitask objective:

$$L_{foundation}=L_{PIE}+0.25L_{LFC\ Huber}+0.20L_{distribution\ Huber}.$$

Dense LFC labels are clipped to `[-4, 4]` with Huber delta 0.5. The distribution term supervises the softmax-weighted bin-center expectation with Huber delta 0.25. This permits LFC-only studies to train the model even when significance or distribution labels are unavailable. Missing distribution observations do not contribute donor counts in the evidence cache.

## Source generalization adaptation

Source amplitude and bulk adaptation use `local_wilcoxon` and `local_cd4` rows. Targets with `SHA256("source-generalization-" + target)[:8] mod 5 == 0` form a fixed validation partition, with up to 16 rows sampled per lineage. This is an adaptation holdout; it is not an independent holdout from foundation pretraining.

For each source sample, every representation of its cell lineage is removed from the response-evidence donors. Context-response inputs are zeroed and context-text tokens are omitted. Other-lineage perturbation evidence remains available. This approximates the controls-only query setting without treating the system as independent of response evidence.

### Arithmetic amplitude

All model parameters are updated from the foundation checkpoint for 1,200 steps. Head learning rate is `1e-5`, backbone learning rate is `2e-6`; batches contain four targets and up to 1,024 shared genes.

$$L_{amplitude}=L_{PIE}+0.50L_{significant\ NMAE}+0.25L_{LFC\ Huber}+0.20L_{distribution\ Huber}.$$

Significant-gene normalized absolute error excludes the perturbed gene and requires at least ten eligible genes. During training, label values are clipped to `[-4, 4]` and the mean absolute true-effect denominator is floored at 0.1. Validation uses untruncated labels and a 0.01 denominator floor. Checkpoints are compared every 300 steps by the mean of lineage-level validation NMAEs.

### Pooled response

The bulk branch starts separately from the foundation checkpoint. Only `lfc_head` and `evidence_output_adapters.lfc` are trained; the model stays in evaluation mode so backbone dropout is disabled. The recipe uses 1,200 steps, batches of eight targets and 1,024 genes, AdamW learning rate `5e-5` and weight decay 0.01.

For pooled control fractions $c$ and predicted LFC $b$, the selected-gene prediction is

$$p_g=c_g2^{clip(b_g,-1.5,1.5)},\qquad p\leftarrow p\frac{\sum_g c_g}{\sum_g p_g}.$$

Response deltas are measured in `log1p(CPM50K)` space relative to controls. The loss is

$$L_{bulk}=MSE(\hat d,d)+0.10(1-\cos(\hat d,d))+0.02L_{InfoNCE}.$$

InfoNCE compares targets within the same context, using cosine similarities and temperature 0.15. The own target gene and unavailable genes are excluded. At validation, select the lowest MSE checkpoint only if cosine and target retrieval each remain within 0.005 of their baseline. The historical selected bulk checkpoint was step 900.

## Specialist amplitude

The specialist addresses a narrower question: can additional nearby-background data improve the amplitude actually used at deployment while retaining source performance?

The deployed effect uses a frozen direction model:

$$\hat a_g=s_g\min(|h_g|,1.5),\quad s_g=\begin{cases}1&b_g\ge0\\-1&b_g<0.\end{cases}$$

The specialist starts from the source-amplitude checkpoint and updates only its LFC readout and LFC evidence adapter. The base direction, significance ranking, pooled-response branch and backbone remain fixed. The historical adaptation panel used 25 MOLM13 targets: 19 training and 6 held out for selection. Those held-out targets are also removed from the specialist's source-training batches.

Measured arithmetic LFCs and significance are computed against a distinct real-control reference. The DE settings are Wilcoxon, arithmetic CPM1M means, FDR within each perturbation, and reference mean at least 5 CPM. The own target gene is excluded. Uncertain labels receive precision weight

$$w_g=\left(1+\frac{SE(\log_2 FC_g)^2}{0.25^2}\right)^{-1},$$

where a delta-method standard error uses the target/control CPM variances and cell counts. Significant genes receive normalized L1 supervision plus weighted Huber loss; nonsignificant measured genes receive a smaller weighted Huber term. Teacher retention penalizes changes to the raw source-amplitude output:

$$L_{specialist}=L_{significant\ NMAE}+0.25L_{significant\ Huber}+0.10L_{nonsignificant\ Huber}+0.02\|h-h_{teacher}\|_2^2.$$

Training runs for 600 steps at learning rate `2e-5`. Every fourth step, beginning at step one, uses two near-background targets on their full modeled gene axis. The other steps use four source targets and up to 2,048 genes. The original selected checkpoint was step 500.

A checkpoint is eligible only when near-background NMAE improves by at least 0.5% and source NMAE rises by at most 0.2%. Among eligible checkpoints, minimize `0.75 × source error ratio + 0.25 × near error ratio`.

## Control-based routing

The router compares each query's pooled control profile with source and fallback-control prototypes. Pearson correlation is computed on `log1p(CPM50K)` profiles, using a fixed gene gate where at least one reference has pooled fraction at least `1e-5`.

The MOLM13 specialist is used only when its prototype is nearest. Every other nearest prototype retains the source-amplitude branch. The historical rule routed D to the specialist and kept E/F on their previous branch. The reference set included source backgrounds and THP1/U87 proxy controls; it did not route individual targets by their measured responses.

This routing was an exploratory change after seeing different transfer behavior across local panels. Its performance should not be described as a blinded routing evaluation.

## Shared-response correction

Source perturbations define a low-rank basis in `log1p(CPM50K)` delta space. The own target gene is set to zero; adaptation-validation targets are excluded. Source rows are weighted by the inverse square root of their lineage's row count. An uncentered randomized SVD fits up to 128 components; V20 uses the first 64 with correction strength 0.5.

Let $d_t$ be the predicted delta for target $t$, $\bar d$ its average over the complete target panel, and $U$ the basis. Apply

$$d'_t=d_t+\alpha(UU^\top\bar d-\bar d),\qquad\alpha=0.5.$$

Every target receives the same correction. Therefore $d'_t-d'_s=d_t-d_s$ before the subsequent nonlinear steps. This avoids projecting every target-specific response into the source subspace. For example, deltas `(0.4, 0.2)` and `(0.1, -0.1)` keep their difference `(0.3, 0.3)` after any shared shift.

Reconstruction applies `expm1`, nonnegativity, bounds of `control × 2^[−1.5, 1.5]`, control fallback for unmodeled genes, and 0.3 retention for the perturbed gene. Profiles are normalized to sum to one. These nonlinear operations can change pairwise contrasts; preservation applies to the preceding shared-shift operation.

The complete target panel is part of this calculation. Rendering a subset of targets still computes the shared shift from the full target list.

## Single-cell count generation

The renderer selects 400 representative controls using library-size strata and the leading principal component of the 2,000 most variable log-normalized genes. If fewer than 400 controls exist, sampling uses replacement with fixed target-specific seeds. Historical selection arrays can be supplied to avoid SVD implementation differences.

The base significance head ranks eligible genes. Up to 4,000 genes receive shape adjustments using frozen direction and adapted magnitude, clipped to `[0.01, 1.5]`. Strength decays linearly from 1.0 to 0.15 over the first 300 ranks. Remaining modeled effects above 0.01 receive a weaker tail adjustment, capped at 0.25. Own-target counts are thinned with retention probability 0.3.

The count kernels re-pair and round counts to the baseline bulk totals, then apply three readout-repair passes with at most 24 moves per repair operation and fraction 0.35. The historical worker seed is `9000 + original_target_index`.

Finally, integer transport replaces the baseline gene totals with the corrected bulk totals while preserving each generated cell's library size. It removes mass from surplus columns and redistributes it into deficit columns subject to row capacity. The algorithm achieves the L1 lower bound

$$\|Y-X\|_1=\|T_{new}-T_{old}\|_1.$$

For example, source cells `[[2, 0, 0], [4, 0, 0]]` can be moved to gene totals `[0, 3, 3]` without changing cell libraries `[2, 4]`. This minimizes count movement; it does not establish that the resulting single-cell distribution is biologically unique.
