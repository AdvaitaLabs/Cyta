# Data and model assets

## Release boundary

The repository ships code, configurations and fingerprints. It does not ship trained checkpoints, expression matrices, cached evidence, gene embeddings, exact historical split files, representative control selections or final prediction packages. No public checkpoint download is claimed. The synthetic demo is self-contained; trained-model execution is not.

All asset paths are supplied by configuration. Keep assets in the ignored `data/`, `checkpoints/` and `artifacts/` directories, or use absolute paths to external storage. `configs/assets.example.json` lists the historical source and dataset names. A shorter source list is not interchangeable with an existing checkpoint: its source dimensions and data statistics must match.

## Required model inputs

| Asset | Role |
| --- | --- |
| Foundation checkpoint | Significance and direction; initialization for adaptation |
| Source-amplitude checkpoint | Default arithmetic effect magnitude |
| Contrastive-bulk checkpoint | Pooled population response |
| Specialist-amplitude checkpoint | Routed near-background magnitude |
| Source preprocessed directories | Response-evidence donors and training records |
| Source token directories | Biological knowledge representations expected by the checkpoint |
| Gene-query text embeddings | Canonical gene-query axis and vectors |
| Split directory and aliases | Exact training-record partition and identifier resolution |
| Common-response basis | Source-derived rank-64 bulk correction |
| Router reference profiles | Background selection using control expression |
| Control selection arrays | Optional historical 400-cell selections for count parity |

The retained checkpoint hashes are listed in [`artifact_fingerprints.json`](../configs/artifact_fingerprints.json). Verify transferred files against them before attempting historical reproduction. New training runs create different artifacts and must be evaluated as new models.

## Source corpus

The foundation corpus combines PIE-preprocessed Replogle/Nadig, Tahoe, Jiang, Arc VCC 2025 and X-Atlas Orion datasets with local harmonized records named `local_wilcoxon`, `local_cd4`, `local_bulk`, `extra_profiles` and `extra_pooled`. These local names describe prepared artifacts, not standalone public dataset accession IDs.

Dataset row counts, gene counts, context counts and metadata hashes from the retained manifest are in [`source_manifest.json`](../configs/source_manifest.json). Counts describe the prepared directories; deduplication and split eligibility produce the actual training-row count.

The source-token modalities are `esm2`, `ncbi_text`, `string_space`, `depmap_gene_effect`, `context_text`, `perturbation_text`, `smiles`, `l1000_tas`, `prism_secondary` and `jump_morphology`. Query inference omits context-text tokens. These are precomputed representations; this repository does not require calling an external language-model API during prediction.

Start with the [PIE repository](https://github.com/ArcInstitute/pie) and its [data/model collection](https://huggingface.co/collections/arcinstitute/pie) for provider documentation and asset formats. Historical local harmonization and exact split membership cannot be reconstructed from the directory counts alone. Reusing a trained checkpoint requires its matching prepared assets; training on newly prepared sources is a new experiment.

## Query controls and target list

Each context uses an H5AD file with cells as rows, genes as columns, and **raw counts in `X`**. Gene identifiers must be nonempty, unique after uppercasing, and in the desired final output order. Every cell must have a positive library. Normalized, log-transformed or negative expression is rejected.

`pert_counts.csv` contains a `target_gene` column in a fixed order. That ordering determines worker seeds and the common-response panel. A separate `gene_names.csv` has a `gene_name` column for final output validation.

All contexts, source bulk archives, response bases and router references must share the full output gene axis where required. Align and filter upstream count datasets explicitly before preparation; do not relabel different genes as if they were the same feature.

## Portable readout archives

Inference writes an NPZ archive for each role and context:

| Key | Shape | Meaning |
| --- | --- | --- |
| `genes` | `(G,)` | Modeled canonical gene identifiers |
| `targets` | `(T,)` | Predicted target identifiers |
| `lfc` | `(T, G)` | Continuous log2 fold changes |
| `p_de` | `(T, G)` | DE probabilities in `[0, 1]` |
| `delta_p` | `(T, G)` | Distribution-change readout |
| `control_mean_cpm` | `(G,)` | Arithmetic per-cell CPM1M mean |
| `control_pool_fraction` | `(G,)` | Pooled control fractions |

The renderer consumes `lfc` from each branch and `p_de` from the base branch. Each branch must cover the same modeled genes. Target rows are aligned by identifier; missing targets fail explicitly. NPZ loading disables pickle.

## Preparing pooled source supervision

`advaita prepare-source --config configs/source.example.json` reads a count H5AD for each context. `obs.target_gene` identifies targets and controls, with configurable column/control names. The example includes one context; add every required source context for a complete training recipe. Context names must match the PIE preprocessed metadata.

The input must already have the common output gene axis. Targets with fewer than `min_cells` measured cells are excluded. The archive fields are:

- `genes`, `targets`: explicit axes.
- `pool`: `(T, G)` pooled target fractions.
- `control`: `(T, G)` pooled control fractions repeated for each target.
- `features`: `(T, G)` mean per-cell `log1p(CPM10K)` control features.
- `mean`: `(T, G)` arithmetic target fractions.
- `counts`: `(T,)` measured cells per target.

For large source collections, create context-specific H5AD inputs before running this command; source preparation reads one context into memory. Local source gene features and response labels must be prepared under the original measurement and filtering rules for checkpoint parity.

## Preparing specialist labels

`advaita prepare-specialist` takes three H5AD inputs: adaptation perturbation cells, independent reference controls, and query-feature controls. Their gene axes must match and globally unique cell identifiers must not overlap. In addition, keep adaptation perturbation cells disjoint from scored validation perturbation cells when constructing the panel; the command cannot verify a held-out file it was not given.

Provide a target split CSV:

```csv
target_gene,split
TARGET_A,train
TARGET_B,train
TARGET_C,validation
```

The preparation command writes `lfc`, `significant`, `tested`, `precision`, `is_validation` and control `features`, together with the gene and target axes. It uses the original cell-eval Wilcoxon arithmetic-mean settings and delta-method precision weighting. The specialist recipe is designed around the MOLM13 adaptation background; adapting another specialist also requires updating the lineage/context convention and reference prototypes.

## Basis and router

`advaita fit-basis` excludes adaptation-validation targets, balances source lineages, and fits the configured low-rank response basis. Keep final-test responses and local evaluation labels outside these inputs.

`advaita fit-router` accepts control-only H5AD files or source NPZ files containing `genes` and `control`. A two-dimensional `control` array is averaged over its target axis. Add all source references needed by the intended recipe; the example router lists only a small subset. The full reference set matters because routing uses the nearest profile.

External data and representations retain their original terms. Obtain them directly from the relevant providers and record their revisions, preprocessing and hashes alongside each experiment.
