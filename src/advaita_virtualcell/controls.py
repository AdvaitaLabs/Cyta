"""Control statistics and representative cell selection."""

import h5py
import numpy as np
from scipy import sparse

from .io import validate_counts


class CountMatrixReader:
    """Read raw H5AD count rows without caching entire sparse data arrays."""

    def __init__(self, path):
        self.handle = h5py.File(path, "r")
        self.matrix = self.handle["X"]
        self.dense = isinstance(self.matrix, h5py.Dataset)
        if self.dense:
            self.shape = tuple(self.matrix.shape)
            self.dtype = self.matrix.dtype
        else:
            if self.matrix.attrs.get("encoding-type") != "csr_matrix":
                self.close()
                raise ValueError("Backed control reading requires dense or CSR H5AD counts")
            self.shape = tuple(self.matrix.attrs["shape"])
            self.dtype = self.matrix["data"].dtype
            self.pointers = self.matrix["indptr"][:]

    def __getitem__(self, selection):
        rows = np.arange(self.shape[0])[selection]
        rows = np.atleast_1d(rows)
        if self.dense:
            if isinstance(selection, slice):
                return sparse.csr_matrix(self.matrix[selection])
            return sparse.csr_matrix(np.stack([self.matrix[int(row)] for row in rows]))
        if not len(rows):
            return sparse.csr_matrix((0, self.shape[1]), dtype=self.dtype)
        if len(rows) == 1 or np.all(np.diff(rows) == 1):
            start, end = int(rows[0]), int(rows[-1]) + 1
            first, last = int(self.pointers[start]), int(self.pointers[end])
            return sparse.csr_matrix(
                (
                    self.matrix["data"][first:last],
                    self.matrix["indices"][first:last],
                    self.pointers[start : end + 1] - first,
                ),
                shape=(len(rows), self.shape[1]),
            )
        values, indices, pointers = [], [], [0]
        for row in rows:
            first, last = int(self.pointers[row]), int(self.pointers[row + 1])
            values.append(self.matrix["data"][first:last])
            indices.append(self.matrix["indices"][first:last])
            pointers.append(pointers[-1] + last - first)
        return sparse.csr_matrix(
            (np.concatenate(values), np.concatenate(indices), np.asarray(pointers)),
            shape=(len(rows), self.shape[1]),
        )

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def control_statistics(counts):
    matrix = validate_counts(counts).astype(np.float64)
    libraries = np.asarray(matrix.sum(1)).ravel()
    fractions = sparse.diags(1.0 / libraries) @ matrix
    arithmetic = np.asarray(fractions.mean(0)).ravel()
    pooled = np.asarray(matrix.sum(0)).ravel()
    pooled /= pooled.sum()
    normalized = sparse.diags(1e4 / libraries) @ matrix
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    features = np.asarray(normalized.mean(0)).ravel()
    return {"arithmetic": arithmetic, "pooled": pooled, "features": features}


def render_control_statistics(counts, chunk_size=1024):
    """Replay the renderer's input precision and its two pooled-count reductions.

    The original renderer accumulated columns in the input dtype. The common
    response stage summed 1024-cell blocks into float64. Keep both reductions:
    converting a float32 control matrix to float64 first can change integer counts.
    """
    n_cells, n_genes = counts.shape
    if n_cells < 1 or n_genes < 1:
        raise ValueError("Controls must be nonempty")
    dtype = np.dtype(counts.dtype)
    sum_dtype = np.asarray(np.zeros(1, dtype=dtype).sum()).dtype
    mean_dtype = np.dtype(np.float64) if dtype.kind in "iu" else dtype
    pooled = np.zeros(n_genes, sum_dtype)
    arithmetic = np.zeros(n_genes, mean_dtype)
    chunked = np.zeros(n_genes, np.float64)
    for start in range(0, n_cells, chunk_size):
        block = validate_counts(counts[start : start + chunk_size]).tocsr()
        libraries = np.asarray(block.sum(1)).ravel()
        np.add.at(pooled, block.indices, block.data)
        fractions = (sparse.diags(1 / libraries) @ block).astype(mean_dtype)
        fractions.data *= np.asarray(1 / n_cells, dtype=mean_dtype)
        np.add.at(arithmetic, fractions.indices, fractions.data)
        chunked += np.asarray(block.sum(0)).ravel()
    pooled = pooled.astype(float)
    return {
        "arithmetic": arithmetic,
        "pooled": pooled / pooled.sum(),
        "projection_pooled": chunked / chunked.sum(),
    }


def select_control_cells(counts, n_targets, cells_per_target=400):
    matrix = validate_counts(counts).astype(np.float64)
    libraries = np.asarray(matrix.sum(1)).ravel()
    n_cells = matrix.shape[0]
    k = int(cells_per_target)
    if k < 2 or n_targets < 1:
        raise ValueError("At least two output cells and one target are required")
    if n_cells < k:
        return np.stack(
            [
                np.random.default_rng(17 + i).choice(n_cells, k, replace=True)
                for i in range(n_targets)
            ]
        )
    if n_cells == k:
        return np.tile(np.arange(n_cells), (n_targets, 1))
    normalized = (sparse.diags(1e4 / libraries) @ matrix).log1p()
    mean = np.asarray(normalized.mean(0)).ravel()
    variance = np.asarray(normalized.multiply(normalized).mean(0)).ravel() - mean**2
    selected = np.argsort(-variance)[:2000]
    centered = normalized[:, selected].toarray() - mean[selected]
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vectors[0]
    strata = np.array_split(np.argsort(libraries, kind="stable"), k)
    selections = []
    for target_index in range(n_targets):
        rows = []
        for stratum_index, indices in enumerate(strata):
            indices = indices[np.argsort(pc1[indices], kind="stable")]
            rows.append(indices[(target_index * 7 + stratum_index) % len(indices)])
        selections.append(np.sort(np.asarray(rows)))
    return np.stack(selections)
