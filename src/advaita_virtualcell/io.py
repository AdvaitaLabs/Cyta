"""Array contracts, count validation, and atomic metadata writes."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_sha256(matrix):
    data = np.ascontiguousarray(sparse.csr_matrix(matrix).toarray(), dtype=np.int32)
    return hashlib.sha256(data.tobytes()).hexdigest()


def canonical_names(values, name="names"):
    values = np.asarray(values, dtype=str)
    if values.ndim != 1 or not len(values) or np.any(np.char.str_len(values) == 0):
        raise ValueError(f"{name} must be a nonempty one-dimensional string axis")
    canonical = np.char.upper(values)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} contains duplicate case-insensitive identifiers")
    return canonical


def validate_counts(matrix, *, positive_libraries=True, max_library=None):
    matrix = sparse.csr_matrix(matrix)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Counts must be a nonempty cells-by-genes matrix")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError("Raw integer counts are required; normalized expression is not accepted")
    if matrix.data.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Count values exceed int32 capacity")
    libraries = np.asarray(matrix.sum(axis=1)).ravel()
    if positive_libraries and np.any(libraries <= 0):
        raise ValueError("Every cell must have a positive library size")
    if max_library is not None and np.any(libraries > max_library):
        raise ValueError("A cell exceeds the configured library-size limit")
    return matrix


def load_readouts(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: archive[key] for key in ("genes", "targets", "lfc", "p_de") if key in archive.files
        }
    for key in ("genes", "targets", "lfc"):
        if key not in data:
            raise ValueError(f"Readout archive is missing {key}: {path}")
    canonical_names(data["genes"], "genes")
    canonical_names(data["targets"], "targets")
    shape = (len(data["targets"]), len(data["genes"]))
    for key in ("lfc", "p_de"):
        if key in data and (data[key].shape != shape or not np.isfinite(data[key]).all()):
            raise ValueError(f"Invalid {key} readout shape or nonfinite value")
    if "p_de" in data and np.any((data["p_de"] < 0) | (data["p_de"] > 1)):
        raise ValueError("DE probabilities must be in [0, 1]")
    return data


def align_readouts(data, genes, targets, field="lfc", dtype=np.float64):
    gene_lookup = dict(zip(canonical_names(data["genes"]), range(len(data["genes"]))))
    target_lookup = dict(zip(canonical_names(data["targets"]), range(len(data["targets"]))))
    columns = np.array([gene_lookup.get(g, -1) for g in canonical_names(genes)])
    requested = canonical_names(targets, "targets")
    missing = [t for t in requested if t not in target_lookup]
    if missing:
        raise ValueError(f"Readouts are missing requested targets: {missing[:5]}")
    rows = np.array([target_lookup[t] for t in requested])
    available = columns >= 0
    output = np.zeros((len(targets), len(genes)), dtype=dtype)
    output[:, available] = data[field][rows][:, columns[available]]
    return output, available
