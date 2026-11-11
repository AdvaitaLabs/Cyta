# Evaluation and limitations

## Local panel design

Use query controls for model features and a separate real-control reference for scoring. Perturbed cells used for adaptation must be disjoint from scored perturbation cells. Keep a fixed target list, gene axis and cell cap, and record which targets satisfy each metric's eligibility rules.

The evaluation command reads context-labeled truth and prediction H5ADs. It caps real perturbation groups at 400 cells and samples the same number of predicted cells without replacement, using a deterministic seed derived from the full target list. It does not duplicate scarce real cells or fabricate additional replicates. A target with fewer than two real cells is omitted; each context requires at least two eligible targets for PDS.

## Reported components

| Component | Implementation and interpretation |
| --- | --- |
| NMAE | `cell_eval2.metrics.de.de_lfc_nmae`; arithmetic LFC error on eligible DE genes |
| REACH | `de_direction_reach`, adjudicated universe, `corrected=False` |
| PDS | `discrimination_score`, cosine distance, real controls, `n-1` rank denominator, midrank ties and panel-target exclusion |
| Raw bulk MSE | Squared difference between pooled `log1p(CPM50K)` profiles over the full gene axis, excluding the own target gene |

DE uses the `gpudge` backend with arithmetic CPM1M means, epsilon `1e-9`, no clipping, FDR per perturbation and reference-expression gate at 5 CPM. The same reference and settings are used for real and predicted cells.

`metrics.csv` contains per-target results. `summary.json` reports per-context means and eligible-target counts. Missing DE metrics remain missing. Raw bulk MSE is not the challenge's noise-corrected normalized MSE. The software does not turn these components into an estimated official score.

## Interpretation

Control-profile similarity suggests a relevant cellular background; it does not disclose the official identity of an anonymized test context. Source-target holdouts during adaptation may overlap foundation training. They diagnose adaptation retention, not fully unseen-perturbation generalization.

The historical near-background specialist improved arithmetic-response diagnostics on MOLM13-related local panels but did not transfer uniformly across backgrounds. The control-based specialist rule was added after observing this difference. Local validation panels were reused across development rounds, so apparent gains require confirmation on a fresh panel or external test set.

Scoring a small number of significant targets can produce unstable NMAE and REACH estimates. Repeated prediction subsampling measures sampling variability and must not be presented as independent experimental replication. Compare methods on paired targets; report uncertainty and target counts when drawing conclusions.

The numerical renderer imposes pooled-response and library-size constraints while modifying an empirical control population. Those constraints do not guarantee realistic joint expression, cell-state transitions, perturbation efficiency or a unique response distribution. Poorly matched source evidence and low control coverage remain failure modes.

The repository makes no claim about a publicly visible final-test leaderboard score. Historical submission acceptance, local metric gains and final-test performance are different pieces of evidence.
