"""Separate pooled-response prediction and common-component correction."""

import numpy as np

from .io import canonical_names


def pooled_profiles(control, lfc, genes, targets, *, cap=1.5, target_retention=0.3):
    control = np.asarray(control, dtype=float)
    if control.shape != (len(genes),) or not np.isfinite(control).all() or np.any(control < 0):
        raise ValueError("Invalid pooled control profile")
    if control.sum() <= 0 or not np.isclose(control.sum(), 1.0):
        raise ValueError("Pooled control profile must sum to one")
    if lfc.shape != (len(targets), len(genes)) or not np.isfinite(lfc).all():
        raise ValueError("Invalid bulk LFC matrix")
    lookup = dict(zip(canonical_names(genes), range(len(genes))))
    result = control[None] * np.exp2(np.clip(lfc, -cap, cap))
    for row, target in enumerate(canonical_names(targets, "targets")):
        own = lookup.get(target, -1)
        if own >= 0:
            result[row, own] = control[own] * target_retention
    return result / result.sum(1, keepdims=True)


def correct_common_response(
    profiles, control, genes, targets, available, basis, *, alpha=0.5, cap=1.5, target_retention=0.3
):
    control = np.asarray(control, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[0] != len(genes) or not np.isfinite(basis).all():
        raise ValueError("Basis must be a finite genes-by-rank matrix")
    if not 0 <= alpha <= 1:
        raise ValueError("Common-response correction strength must be in [0, 1]")
    if (
        profiles.shape != (len(targets), len(genes))
        or np.any(profiles < 0)
        or not np.isfinite(profiles).all()
    ):
        raise ValueError("Invalid pooled response profiles")
    available = np.asarray(available, dtype=bool)
    if available.shape != (len(genes),):
        raise ValueError("Gene availability must match the full gene axis")
    lookup = dict(zip(canonical_names(genes), range(len(genes))))
    own = [lookup.get(target, -1) for target in canonical_names(targets, "targets")]
    reference = np.log1p(5e4 * control)
    delta = np.log1p(5e4 * profiles) - reference
    for row, index in enumerate(own):
        if index >= 0:
            delta[row, index] = 0
    center = delta.mean(0)
    shift = alpha * ((center @ basis) @ basis.T - center)
    result = np.maximum(np.expm1(reference[None] + delta + shift[None]), 0) / 5e4
    result = np.clip(result, control * 2**-cap, control * 2**cap)
    result[:, ~available] = control[None, ~available]
    for row, index in enumerate(own):
        if index >= 0:
            result[row, index] = control[index] * target_retention
    result /= result.sum(1, keepdims=True)
    return result, shift


def load_basis(path, genes):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(canonical_names(archive["genes"]), canonical_names(genes)):
            raise ValueError("Response basis and output gene order do not match")
        return archive["basis"], float(archive["alpha"])
