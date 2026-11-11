# Inference and count export

## Execution modes

The code supports trained-model inference and readout-only rendering. Inference requires the `model` extra, the four checkpoints and matching PIE assets. Rendering existing response archives requires only the core dependencies.

All JSON paths resolve relative to the configuration file. Output directories must be empty; existing runs are not silently overwritten. The package does not contain credentials or perform account login or automatic upload.

## Model inference

Run `advaita infer --config configs/infer.example.json` after supplying assets. Control features are mean per-cell `log1p(CPM10K)` expression. Query context identifiers are anonymous and context-text tokens are omitted. Source response evidence is built from the configured training data.

`device: "auto"` uses CUDA when available and CPU otherwise. The default precision is bfloat16 autocast on CUDA and float32 on CPU. CPU and GPU predictions need not be bitwise identical. Source evidence construction, multiple model branches and full-axis queries can require substantial memory; use an environment sized for the original data and model rather than the small demo.

The output contains `base/`, `amplitude/`, `bulk/` and `specialist/` NPZ readouts plus a manifest with checkpoint and prediction hashes. A response archive is not yet a submission matrix.

## Count rendering

Run `advaita render --config configs/render.example.json --workers 1`. Every context receives a background route from controls only. A missing specialist archive is an error when the route requests that specialist.

The renderer preserves the original V20 precision distinctions: baseline pooled totals use input-dtype accumulation, while common-response control pools use 1,024-cell blocks accumulated into float64. Readout alignment, clipping and integer rounding also retain the historical branch behavior. Converting control matrices to a different dtype can change the result even when their integer values are equal.

Cached control selections have shape `(number_of_targets, cells_per_target)` and contain row indices into the exact control file. Remove `control_sets` to recompute representative selections. Recomputing the selection uses a full control-matrix SVD calculation and can use considerably more memory than rendering from cached selections. Backed control access and streamed summary statistics reduce memory when selections are supplied.

Increase `--workers` only after checking memory use; each worker creates dense temporary arrays for one target. For a small production smoke test:

```bash
advaita render --config configs/render.example.json --indices 0 100 200 --workers 1
```

Use a distinct output directory for this partial run. Original target indices and the full target panel are retained, so a partial render uses the same seeds and common shift as a complete run. Its manifest intentionally contains only the selected blocks. It is not a complete final-test artifact.

## Assembly and checks

```bash
advaita assemble --manifest outputs/counts/manifest.json \
  --output outputs/Advaita_VirtualCell_Model.h5ad
advaita validate --input outputs/Advaita_VirtualCell_Model.h5ad \
  --targets data/final/pert_counts.csv --genes data/final/gene_names.csv \
  --contexts D E F --cells-per-target 400
```

Assembly verifies block identities, dimensions, canonical sparse storage and count hashes before writing. It then streams CSR arrays into H5AD. The final validation independently checks the requested context-target coverage and full gene order, in addition to positive libraries and finite nonnegative integer counts. Always provide the complete target list and expected contexts when validating a final-test matrix; otherwise a structurally valid subset can pass a shape-only check.

The H5AD layout is:

- `X`: sparse CSR integer RNA counts.
- `obs.context`: query background identifier.
- `obs.target_gene`: perturbed target identifier.
- `obs_names`: unique context/target/cell identifiers.
- `var_names`: full requested gene axis.
- `uns.model_name`: `Advaita_VirtualCell_Model`.

A separate `.validation.json` records the assembled file's hash and dimensions.

## Challenge packaging

This repository exports and validates H5AD predictions. The challenge's CLI remains responsible for its current schema, packaging, account authorization and submission. Use the official final/test workflow and the CLI version required by the organizer; validation and final submissions are different tracks.

Check the challenge's current instructions and the installed CLI's `--help` before packaging. This project does not embed historical account IDs, upload commands or receipts. Successful local count validation does not itself establish that a submission was accepted or scored by the organizer.
