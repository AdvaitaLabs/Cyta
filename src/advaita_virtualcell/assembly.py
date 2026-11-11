"""Bounded-memory assembly of validated sparse blocks into AnnData."""

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from . import MODEL_NAME
from .io import count_sha256, file_sha256, read_json, validate_counts, write_json


def assemble(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    if not manifest.get("complete"):
        raise ValueError("Rendering manifest is incomplete")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = manifest["blocks"]
    if not records:
        raise ValueError("No count blocks were provided")
    genes = np.asarray(manifest["genes"], str)
    cells = int(manifest["settings"]["cells_per_target"])
    expected = {
        (context, index)
        for context in manifest["contexts"]
        for index in manifest["selected_indices"]
    }
    seen = set()
    obs_parts = []
    paths = []
    nnz = 0
    for record in records:
        key = (record["context"], record["index"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected context-target block")
        seen.add(key)
        if record["target"] != manifest["targets"][record["index"]]:
            raise ValueError("Block target does not match the manifest ordering")
        path = root / record["context"] / record["path"]
        matrix = validate_counts(sparse.load_npz(path))
        if matrix.shape != (cells, len(genes)) or not matrix.has_canonical_format:
            raise ValueError("Invalid block shape or sparse format")
        if count_sha256(matrix) != record["counts_sha256"]:
            raise ValueError(f"Count block integrity check failed: {path.name}")
        paths.append(path)
        nnz += matrix.nnz
        obs_parts.append(
            pd.DataFrame(
                {"context": record["context"], "target_gene": record["target"]},
                index=[f"{record['context']}_{record['index']:04d}_{i:04d}" for i in range(cells)],
            )
        )
    if seen != expected:
        raise ValueError("The manifest is missing expected context-target blocks")
    obs = pd.concat(obs_parts)
    shape = (len(obs), len(genes))
    temporary = output.with_suffix(".partial.h5ad")
    data = ad.AnnData(
        sparse.csr_matrix(shape, dtype=np.int32), obs=obs, var=pd.DataFrame(index=genes)
    )
    data.uns["model_name"] = MODEL_NAME
    data.write_h5ad(temporary)
    pointers = np.empty(shape[0] + 1, np.int64)
    pointers[0] = 0
    offset = row = 0
    with h5py.File(temporary, "a") as handle:
        del handle["X"]
        group = handle.create_group("X")
        group.attrs.update(
            {"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": shape}
        )
        values = group.create_dataset(
            "data",
            (nnz,),
            dtype=np.int32,
            chunks=(min(1 << 20, nnz),),
            compression="gzip",
            compression_opts=1,
        )
        columns = group.create_dataset(
            "indices",
            (nnz,),
            dtype=np.int32,
            chunks=(min(1 << 20, nnz),),
            compression="gzip",
            compression_opts=1,
        )
        for path in paths:
            matrix = sparse.load_npz(path)
            end = offset + matrix.nnz
            values[offset:end] = matrix.data
            columns[offset:end] = matrix.indices
            pointers[row + 1 : row + len(matrix.indptr)] = matrix.indptr[1:] + offset
            row += matrix.shape[0]
            offset = end
        if row != shape[0] or offset != nnz:
            raise ArithmeticError("Sparse assembly size mismatch")
        group.create_dataset("indptr", data=pointers, compression="gzip", compression_opts=1)
    temporary.replace(output)
    result = {
        "model_name": MODEL_NAME,
        "complete": True,
        "shape": list(shape),
        "blocks": len(records),
        "nnz": nnz,
        "sha256": file_sha256(output),
    }
    write_json(output.with_suffix(".validation.json"), result)
    return result
