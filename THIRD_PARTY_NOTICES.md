# Third-party notices

## PIE

- Project: PIE, by Rishi Verma and the Arc Institute contributors.
- Source: https://github.com/ArcInstitute/pie
- Revision: `038940057545ae686db946ae247aeb0ff0101897`
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
- Original license: https://github.com/ArcInstitute/pie/blob/038940057545ae686db946ae247aeb0ff0101897/LICENSE.md

Advaita_VirtualCell_Model uses the PIE Perceiver architecture, data contracts, source encoders, prediction readouts, evidence representation and multitask losses. PIE remains an external dependency. The evidence cache extensions, missing-label handling and training integration in this repository adapt PIE interfaces and behavior; these adaptations are not the unmodified upstream implementation.

Advaita Lab's additions include source-generalization adaptation, separate arithmetic and pooled-response objectives, specialist amplitude adaptation, background routing, common-response correction, integer count generation, and the accompanying orchestration and documentation. The repository retains the upstream noncommercial share-alike license. No upstream endorsement is implied.

## Evaluation and numerical dependencies

Local DE and discrimination evaluation delegates to the installed `cell-eval2` and `gpudge` packages. NumPy, SciPy, PyTorch, pandas, AnnData, h5py and other dependencies are installed separately and retain their respective licenses. Their source distributions are not vendored in this repository.

## Data and pretrained representations

Expression datasets and biological source embeddings are not included. Obtain them from their original providers and follow their individual access, redistribution, attribution and use requirements. The repository license does not relicense external datasets, embeddings, model weights or challenge data.
