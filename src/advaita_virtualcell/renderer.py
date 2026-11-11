"""Convert response readouts into integer single-cell counts."""

import numpy as np
from scipy import sparse

from ._count_kernels import (
    fix_totals,
    pair_to_totals,
    readout_repair,
    shape_shift_gene,
    shift_gene,
    tail_shift_gene,
)
from .config import RenderConfig
from .io import validate_counts
from .transport import integer_totals, transport_counts


def make_calls(
    base_lfc,
    amplitude_lfc,
    probabilities,
    available,
    control_mean,
    target_index,
    config=None,
    *,
    specialist=False,
):
    config = config or RenderConfig()
    base_lfc = np.asarray(base_lfc)
    signs = np.where(base_lfc >= 0, 1.0, -1.0) if specialist else np.sign(base_lfc)
    effects = np.clip(signs * np.abs(amplitude_lfc), -config.amplitude_cap, config.amplitude_cap)
    eligible = np.asarray(available, bool) & (control_mean > 0) & np.isfinite(effects)
    if target_index >= 0:
        eligible[target_index] = False
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(-probabilities[ranked], kind="stable")]
    selected = ranked[: config.top_de_genes]
    eligible[selected] = False
    dense = np.flatnonzero(eligible & (np.abs(effects) > config.amplitude_floor))
    direction = signs if specialist else np.where(effects >= 0, 1.0, -1.0)
    return (
        selected,
        direction[selected],
        np.clip(np.abs(effects[selected]), config.amplitude_floor, config.amplitude_cap),
        dense,
        direction[dense],
        np.minimum(np.abs(effects[dense]), config.tail_amplitude_cap),
    )


def render_counts(
    base, target_totals, calls, target_index, control_mean, *, target_seed_index=0, config=None
):
    config = config or RenderConfig()
    base = validate_counts(base)
    counts = base.toarray().astype(np.int64)
    libraries = counts.sum(1).astype(np.float64)
    totals = np.asarray(target_totals, np.int64).copy()
    if totals.shape != (counts.shape[1],) or np.any(totals < 0) or totals.sum() <= 0:
        raise ValueError(
            "Target totals must be nonnegative, positive in sum and match the gene axis"
        )
    if target_index < -1 or target_index >= counts.shape[1]:
        raise ValueError("Target gene index is out of bounds")
    rng = np.random.default_rng(config.worker_seed + int(target_seed_index))
    genes, signs, amplitudes = calls[:3]
    sample_mean = (counts / libraries[:, None]).mean(0)
    ratio = np.where(
        sample_mean > 0, np.clip(control_mean / np.maximum(sample_mean, 1e-30), 0.5, 2.0), 1.0
    )
    for gene, sign, amplitude in zip(*calls[3:6], strict=True):
        factor = 2.0 ** (sign * amplitude) * ratio[gene]
        if config.dense_tail > 0:
            counts[:, gene] = tail_shift_gene(counts[:, gene], factor, config.dense_tail, rng)
        else:
            counts[:, gene] = shift_gene(counts[:, gene], factor, rng)
    strengths = 1.0 + (config.rank_floor - 1.0) * np.minimum(
        np.arange(len(genes)) / (config.rank_decay_genes - 1.0), 1.0
    )
    for position, (gene, sign, amplitude) in enumerate(zip(genes, signs, amplitudes, strict=True)):
        counts[:, gene] = shape_shift_gene(
            counts[:, gene],
            libraries,
            sign,
            strengths[position],
            amplitude,
            rng,
            mref=float(sample_mean[gene] * ratio[gene]),
        )
    if target_index >= 0:
        counts[:, target_index] = rng.binomial(counts[:, target_index], config.target_retention)
    called = np.zeros(counts.shape[1], bool)
    called[genes] = True
    if target_index >= 0:
        called[target_index] = True
    result, reachable = pair_to_totals(counts, libraries, totals, rng, mode="stoch")
    result = fix_totals(result, totals, libraries, rng, ~reachable)
    if np.any(result < 0) or not np.array_equal(result.sum(0), totals):
        raise ArithmeticError("Count rendering failed its column constraints")
    info = {
        "calls": len(genes),
        "out_of_capacity": int((~reachable & ~called & (totals != counts.sum(0))).sum()),
        "lib_cv_change": float(np.std(result.sum(1) / libraries)),
    }
    if result.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("Rendered count exceeds int32 capacity")
    result = sparse.csr_matrix(result.astype(np.int32))
    result.eliminate_zeros()
    result.sort_indices()
    if config.repair_passes > 0:
        result, repair = readout_repair(
            result,
            base,
            calls,
            target_index,
            passes=config.repair_passes,
            max_moves=config.repair_max_moves,
            fraction=config.repair_fraction,
            support=True,
            distributed=False,
            moderate=True,
            own_keep=config.target_retention,
            mref=sample_mean * ratio,
        )
        info.update(repair)
    validate_counts(result)
    return result, info


def render_target(
    base,
    control_mean,
    base_lfc,
    amplitude_lfc,
    probabilities,
    available,
    target_index,
    baseline_profile,
    corrected_profile,
    *,
    seed_index,
    config=None,
    specialist=False,
):
    config = config or RenderConfig()
    calls = make_calls(
        base_lfc,
        amplitude_lfc,
        probabilities,
        available,
        control_mean,
        target_index,
        config,
        specialist=specialist,
    )
    total = int(base.sum())
    original_totals = integer_totals(baseline_profile, total)
    corrected_totals = integer_totals(corrected_profile, total)
    generated, rendering = render_counts(
        base,
        original_totals,
        calls,
        target_index,
        control_mean,
        target_seed_index=seed_index,
        config=config,
    )
    result, transport = transport_counts(generated, corrected_totals)
    return result, {"rendering": rendering, "transport": transport, "specialist": bool(specialist)}
