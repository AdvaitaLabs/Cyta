# Reproducibility

## Three levels of reproduction

1. **Software execution:** the synthetic CPU demo verifies rendering, assembly and validation without external assets.
2. **Historical count reproduction:** requires the original response readouts, control files with their original dtype, complete target/gene axes, basis, routing references and control selections. These inputs can reproduce the deterministic count pipeline without rerunning the model.
3. **Model retraining:** additionally requires the complete preprocessed source corpus, exact split membership, source embeddings, aliases and training statistics. Matching hyperparameters alone does not establish identical weights.

The trained assets are not bundled. [`artifact_fingerprints.json`](../configs/artifact_fingerprints.json) identifies the retained foundation, amplitude, bulk, specialist and response-basis files. [`source_manifest.json`](../configs/source_manifest.json) records prepared dataset metadata fingerprints without machine-specific locations.

## Historical runtime

The retained local runtime used Python 3.12.3, PyTorch 2.8.0 with CUDA 12.8, NumPy 2.3.2, SciPy 1.18.1, pandas 3.1.0, AnnData 0.13.4 and h5py 3.16.0. Local DE diagnostics used `cell-eval2` 0.16.0. This is a provenance record, not a promise that a clean package resolver will reproduce that environment.

The public model extra pins PIE revision `038940057545ae686db946ae247aeb0ff0101897`. Its declared dependency bounds are newer than parts of the historical runtime. A supported fresh installation may therefore use different library versions. Numerical changes, CUDA kernels, bfloat16 arithmetic and randomized SVD can affect new readouts or regenerated bases.

For count parity, preserve cached readouts, basis and control selections. For scientifically meaningful retraining, report the new environment and compare resulting predictions on declared validation panels.

## Deterministic behavior

- Foundation seed: 20261014.
- Source-adaptation seed: 20261029.
- Source response-basis seed: 20261030.
- Specialist seed: 20261101.
- Count-worker seed: 9000 plus the original target-list index.
- Small-control resampling seed: 17 plus the original target-list index.

Sorting is stable where target/gene ranks or largest remainders determine discrete counts. Partial renders retain the complete target list when computing the shared response shift. Target order, reference prototype order and gene order are explicit inputs.

Float32 control reductions and float64 common-response reductions are intentionally distinct. Do not cast controls, change sparse accumulation order or regenerate control selections when the goal is historical count identity.

## Validation scope

The public tests cover exact row/column marginals and the L1 movement lower bound; previously absent genes; stable rounding; arithmetic versus pooled estimands; control selection; routing; readout alignment; finite training gradients; source-retention gates; frozen specialist modules; and the synthetic render/assembly pipeline with tamper detection.

Historical count comparisons are performed separately against retained production artifacts. Their scope and results are recorded in [`release_validation.json`](../configs/release_validation.json). Production data and output blocks are not committed as fixtures.

This code organization does not claim that a new full GPU training run was completed as part of the public release. Complete model inference requires the external source assets, and no end-to-end training performance claim follows from the software tests.
