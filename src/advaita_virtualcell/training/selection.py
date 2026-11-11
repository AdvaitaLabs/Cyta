"""Checkpoint selection gates fixed before panel evaluation."""

import math


def specialist_candidate(source, near, baseline_source, baseline_near):
    values = (source, near, baseline_source, baseline_near)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Selection metrics must be finite nonnegative values")
    if baseline_source == 0 or baseline_near == 0:
        raise ValueError("Relative selection requires positive baseline errors")
    source_ratio = source / baseline_source
    near_ratio = near / baseline_near
    return {
        "source_ratio": source_ratio,
        "near_ratio": near_ratio,
        "score": 0.75 * source_ratio + 0.25 * near_ratio,
        "passed": near_ratio <= 0.995 and source_ratio <= 1.002,
    }


def bulk_candidate(mse, cosine, retrieval, baseline_cosine, baseline_retrieval):
    if not all(
        math.isfinite(v) for v in (mse, cosine, retrieval, baseline_cosine, baseline_retrieval)
    ):
        return False
    return cosine >= baseline_cosine - 0.005 and retrieval >= baseline_retrieval - 0.005
