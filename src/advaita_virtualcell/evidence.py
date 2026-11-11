"""Response evidence that excludes missing observations and matching lineages."""

from collections import defaultdict

import numpy as np

from ._evidence_cache import sampled_percentile, streaming_builder, streaming_context_response


def install_evidence_extension():
    from pie.data import datamodule
    from pie.data import evidence as ev

    if ev.EVIDENCE_SCHEMA_VERSION == 4:
        return
    if ev.EVIDENCE_SCHEMA_VERSION != 3:
        raise RuntimeError("This release requires PIE evidence schema 3 or the Advaita schema 4")
    ev.EVIDENCE_SCHEMA_VERSION = 4
    valid_bit = 16
    ev._KNOWN_FLAGS |= valid_bit
    original_own = ev._own_row_stats
    original_unpack = ev._unpack_rows
    original_contributing = ev.contributing_dirs

    def own_stats(*, fc, fdr, delta, tested, rules):
        valid = np.isfinite(delta)
        result = original_own(
            fc=fc, fdr=fdr, delta=np.where(valid, delta, 0.0), tested=tested, rules=rules
        )
        result["dp_cnt"] = valid.astype(np.int64)
        return result

    def pack_rows(own):
        for key in ev.COUNT_ARRAYS:
            if not np.isin(own[key], (0, 1)).all():
                raise ValueError(f"{key} must contain binary row contributions")
        delta = np.ascontiguousarray(own["dp_sum"], dtype=np.float32)
        lfc = np.ascontiguousarray(own["lfc_sum"], dtype=np.float32)
        flags = np.zeros(delta.shape, dtype=np.uint8)
        for key, bit in (
            ("tested_cnt", ev._TESTED),
            ("up_cnt", ev._UP),
            ("down_cnt", ev._DOWN),
            ("lfc_cnt", ev._LFC_VALID),
            ("dp_cnt", valid_bit),
        ):
            flags |= own[key].astype(np.uint8) * bit
        ev._validate_records(delta, lfc, flags)
        if np.any(delta[(flags & valid_bit) == 0] != 0):
            raise ValueError("Missing response labels contributed nonzero values")
        return delta, lfc, flags

    def unpack_rows(delta, lfc, flags):
        result = original_unpack(delta, lfc, flags)
        result["dp_cnt"] = ((flags & valid_bit) != 0).astype(np.int64)
        return result

    def contributing(dirs, rows):
        return [
            directory
            for directory in original_contributing(dirs, rows)
            if directory.dataset not in {"local_bulk", "extra_pooled"}
        ]

    ev._own_row_stats = own_stats
    ev._pack_rows = pack_rows
    ev._unpack_rows = unpack_rows
    ev.contributing_dirs = contributing
    ev.build_evidence = streaming_builder
    ev._build_context_response = streaming_context_response
    datamodule.train_percentile = sampled_percentile


def lineage(context):
    name = str(context).lower()
    for group in ("k562", "rpe1", "jurkat", "hepg2", "hct116", "hek293t", "cd4"):
        if group in name:
            return group
    return "h1" if name in {"h1", "arc_h1", "arc_h1_val"} else name


class LineageEvidence:
    def __init__(self, index):
        self.index = index
        self.excluded = defaultdict(list)
        for name, shard in index.evidence.shards.items():
            for context in shard.context_to_index:
                self.excluded[lineage(context)].append((name, context))

    def blocks(self, group, perturbation, gene_ids):
        from pie.data import evidence as ev

        index = self.index
        evidence = index.evidence
        columns = index.gene_cols[gene_ids]
        available = columns >= 0
        statistics = ev._zero_stats(len(gene_ids))
        by_dataset = {
            key: np.zeros((evidence.n_donor_datasets, len(gene_ids)), dtype=np.int64)
            for key in ev.PER_DATASET_ARRAYS
        }
        row = index.pert_rows.get(perturbation, -1)
        if row >= 0 and available.any():
            selected = columns[available]
            for key in ev.GLOBAL_ARRAYS:
                statistics[key][available] = np.asarray(
                    evidence.arrays[key][row], dtype=statistics[key].dtype
                )[selected]
            for key in ev.PER_DATASET_ARRAYS:
                by_dataset[key][:, available] = np.asarray(
                    evidence.by_ds[key][:, row, :], dtype=np.int64
                )[:, selected]
            for name, context in self.excluded[group]:
                own = ev._query_response_stats(
                    index,
                    dataset=name,
                    context=context,
                    perturbation=perturbation,
                    gene_union_ids=gene_ids,
                )
                ev._subtract_donors(statistics, by_dataset, own, index.ds_slot.get(name, -1))
        if any(np.any(statistics[key] < 0) for key in ev.COUNT_ARRAYS):
            raise ArithmeticError("Evidence subtraction produced a negative donor count")
        blocks = ev._derive_blocks(statistics, by_dataset, evidence.scales)
        blocks.update(ev._derive_context_blocks(ev._zero_stats(len(gene_ids)), evidence.scales))
        return blocks
