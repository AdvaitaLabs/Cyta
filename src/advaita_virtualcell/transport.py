"""Minimal-L1 integer transport with exact row and column marginals."""

import numpy as np
from scipy import sparse

from .io import validate_counts


def allocate(weights, total, capacity):
    capacity = np.asarray(capacity)
    weights = np.asarray(weights, dtype=float)
    if capacity.ndim != 1 or weights.shape != capacity.shape:
        raise ValueError("Weights and capacity must be matching vectors")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Allocation weights must be finite and nonnegative")
    if (
        not np.isfinite(capacity).all()
        or np.any(capacity < 0)
        or np.any(capacity != np.floor(capacity))
    ):
        raise ValueError("Capacity must contain nonnegative integers")
    if int(total) != total or not 0 <= total <= int(capacity.sum()):
        raise ValueError("Requested total exceeds allocation capacity")
    cap = capacity.astype(np.int64, copy=True)
    out = np.zeros(len(cap), np.int64)
    remaining = int(total)
    while remaining:
        active = cap > 0
        values = np.where(active, weights, 0)
        if values.sum() <= 0:
            values = cap.astype(float)
        ideal = values * (remaining / values.sum())
        saturated = active & (ideal >= cap)
        if saturated.any():
            take = cap[saturated]
            out[saturated] += take
            remaining -= int(take.sum())
            cap[saturated] = 0
            continue
        take = np.minimum(np.floor(ideal).astype(np.int64), cap)
        out += take
        cap -= take
        remaining -= int(take.sum())
        if remaining:
            eligible = np.flatnonzero(cap > 0)
            order = eligible[np.argsort(-(ideal - take)[eligible], kind="stable")]
            if remaining > len(order):
                raise ArithmeticError("Largest-remainder allocation exceeded available slots")
            out[order[:remaining]] += 1
            remaining = 0
    return out


def integer_totals(profile, total):
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 1 or not np.isfinite(profile).all() or np.any(profile < 0):
        raise ValueError("Profile must be a finite nonnegative vector")
    if int(total) != total or total < 0:
        raise ValueError("Total must be a nonnegative integer")
    if profile.sum() <= 0:
        if total == 0:
            return np.zeros_like(profile, dtype=np.int64)
        raise ValueError("A positive total requires a positive profile")
    values = profile * (total / profile.sum())
    counts = np.floor(values).astype(np.int64)
    remainder = int(total - counts.sum())
    if not 0 <= remainder <= len(counts):
        raise ArithmeticError("Unstable integer rounding")
    counts[np.argsort(-(values - counts), kind="stable")[:remainder]] += 1
    return counts


def transport_counts(base, target):
    original = validate_counts(base).astype(np.int64).toarray()
    raw_target = np.asarray(target)
    if raw_target.shape != (original.shape[1],):
        raise ValueError("Target totals do not match the gene axis")
    if (
        not np.isfinite(raw_target).all()
        or np.any(raw_target < 0)
        or np.any(raw_target != np.floor(raw_target))
    ):
        raise ValueError("Target totals must be finite nonnegative integers")
    target = raw_target.astype(np.int64)
    old = original.sum(0)
    rows = original.sum(1)
    if target.sum() != old.sum():
        raise ValueError("Row and column totals have different total mass")
    result = original.copy()
    difference = target - old
    for gene in np.flatnonzero(difference < 0):
        result[:, gene] -= allocate(result[:, gene], int(-difference[gene]), result[:, gene])
    capacity = rows - result.sum(1)
    order = np.flatnonzero(difference > 0)
    order = order[np.argsort(-difference[order], kind="stable")]
    for gene in order:
        addition = allocate(original[:, gene], int(difference[gene]), capacity)
        result[:, gene] += addition
        capacity -= addition
    if not np.array_equal(result.sum(0), target) or not np.array_equal(result.sum(1), rows):
        raise ArithmeticError("Integer transport failed its marginal constraints")
    movement = int(np.abs(result - original).sum())
    lower_bound = int(np.abs(difference).sum())
    if movement != lower_bound:
        raise ArithmeticError("Integer transport did not attain the L1 lower bound")
    if result.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("Transported count exceeds int32 capacity")
    info = {
        "minimal_count_movement": True,
        "cell_libraries_preserved": True,
        "bulk_columns_exact": True,
        "l1_counts": movement,
        "minimum_l1_counts": lower_bound,
        "l1_fraction": float(movement / max(rows.sum(), 1)),
        "max_cell_l1_fraction": float(np.max(np.abs(result - original).sum(1) / rows)),
    }
    return sparse.csr_matrix(result.astype(np.int32)), info
