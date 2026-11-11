"""Streaming checks for the assembled cells-by-genes prediction matrix."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .controls import CountMatrixReader
from .io import canonical_names, file_sha256, validate_counts


def validate_h5ad(path, *, targets=None, genes=None, contexts=None, cells_per_target=400):
    data = ad.read_h5ad(path, backed="r")
    try:
        if data.n_obs == 0 or data.n_vars == 0:
            raise ValueError("Predictions must contain cells and genes")
        canonical_names(data.var_names, "genes")
        if not data.obs_names.is_unique:
            raise ValueError("Observation identifiers must be unique")
        if not {"context", "target_gene"}.issubset(data.obs.columns):
            raise ValueError("Observations require context and target_gene columns")
        if genes is not None:
            expected = pd.read_csv(genes)["gene_name"].to_numpy(str)
            if not np.array_equal(data.var_names.to_numpy(str), expected):
                raise ValueError("Output gene identifiers or ordering do not match")
        counts = data.obs.groupby(["context", "target_gene"], observed=True).size()
        if (counts != cells_per_target).any():
            raise ValueError("Unexpected cell count for a context-target pair")
        observed_contexts = sorted(set(data.obs.context.astype(str)))
        expected_contexts = list(contexts) if contexts is not None else observed_contexts
        if set(observed_contexts) != set(expected_contexts):
            raise ValueError("Unexpected or missing contexts")
        if targets is not None:
            expected_targets = pd.read_csv(targets)["target_gene"].astype(str).tolist()
            canonical_names(expected_targets, "targets")
            expected_pairs = {
                (context, target) for context in expected_contexts for target in expected_targets
            }
            if set(counts.index) != expected_pairs:
                raise ValueError("Unexpected or missing context-target pairs")
        minimum, maximum, nnz = None, 0, 0
        with CountMatrixReader(path) as reader:
            for start in range(0, data.n_obs, 400):
                block = validate_counts(reader[start : start + 400])
                sizes = np.asarray(block.sum(1)).ravel()
                minimum = int(sizes.min()) if minimum is None else min(minimum, int(sizes.min()))
                maximum = max(maximum, int(sizes.max()))
                nnz += block.nnz
        return dict(
            valid=True,
            file=Path(path).name,
            shape=list(data.shape),
            contexts=observed_contexts,
            context_target_pairs=len(counts),
            cells_per_target=cells_per_target,
            nnz=nnz,
            library_size_min=minimum,
            library_size_max=maximum,
            sha256=file_sha256(path),
        )
    finally:
        data.file.close()
