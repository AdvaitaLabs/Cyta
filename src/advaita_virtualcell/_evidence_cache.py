"""Bounded-memory response evidence with explicit missing-label validity."""

import mmap
from dataclasses import asdict
from pathlib import Path

import numpy as np


def sampled_percentile(delta_p, rows, percentile):
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return None
    if len(rows) > 512:
        rows = np.sort(np.random.default_rng(20261014).choice(rows, 512, replace=False))
    values = np.abs(np.asarray(delta_p[rows], dtype=np.float32))
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else None


def release(arr):
    if isinstance(arr, np.memmap):
        if arr.mode != "r":
            arr.flush()
        if hasattr(arr._mmap, "madvise"):
            arr._mmap.madvise(mmap.MADV_DONTNEED)


def streaming_context_response(root, bdirs, *, rules, train_json_sha256, chunk):
    from pie.data import evidence as ev

    entries = []
    for slot, b in enumerate(bdirs):
        relative = f"{ev.CONTEXT_DIR}/{slot:04d}"
        path = Path(root) / relative
        path.mkdir(parents=True)
        n, g = (len(b.kept), len(b.genes))
        contexts = sorted({c for c, p in b.donor_keys})
        ci = {c: i for i, c in enumerate(contexts)}
        ids = np.array([ci[c] for c, p in b.donor_keys], np.int64)
        totals = {
            k: np.zeros((len(contexts), g), dtype=np.float64 if k in ev.SUM_ARRAYS else np.int64)
            for k in ev.GLOBAL_ARRAYS
        }
        records = {
            k: np.lib.format.open_memmap(path / f"donor_{k}.npy", mode="w+", dtype=dt, shape=(n, g))
            for k, dt in ev._RECORDS
        }
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            ix = b.kept[start:stop]
            own = ev._own_row_stats(
                fc=b.fc[ix], fdr=b.fdr[ix], delta=b.dp[ix], tested=b.tested[ix], rules=rules
            )
            dp, lf, flags = ev._pack_rows(own)
            for name, value in (("dp", dp), ("lfc", lf), ("flags", flags)):
                if value.any():
                    records[name][start:stop] = value
            values = ev._unpack_rows(dp, lf, flags)
            for k in ev.SUM_ARRAYS:
                np.add.at(totals[k], ids[start:stop], values[k])
            for c in np.unique(ids[start:stop]):
                take = ids[start:stop] == c
                for k, value in values.items():
                    if k not in ev.SUM_ARRAYS:
                        totals[k][c] += value[take].sum(0)
            if start % 1024 == 0:
                for a in records.values():
                    release(a)
                for a in (b.fc, b.fdr, b.dp, b.tested):
                    release(a)
        for a in records.values():
            release(a)
        for k, value in totals.items():
            np.save(
                path / f"{k}.npy", value.astype(np.float64 if k in ev.SUM_ARRAYS else np.uint32)
            )
        meta = {
            "dataset": b.dataset,
            "n_rows": n,
            "n_genes": g,
            "n_contexts": len(contexts),
            "gene_symbols": list(b.genes),
            "contexts": contexts,
            "donor_keys": [list(k) for k in b.donor_keys],
            "train_json_sha256": train_json_sha256,
        }
        ev.write_json(path / ev.SHARD_META, meta)
        entries.append({"dataset": b.dataset, "path": relative})
        print("EVIDENCE SHARD", b.dataset, n, flush=True)
    return {"version": ev.CONTEXT_SECTION_VERSION, "shards": entries}


def streaming_builder(dirs, train_split, train_json_sha256, cfg, fdr_threshold, out):
    from pie.data import evidence as ev

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    rows = ev.resolve_split(train_split, dirs)
    contributing = ev.contributing_dirs(dirs, rows)
    bdirs = [ev._open_build_dir(d, rows[d.dataset]) for d in contributing]
    key = ev.evidence_cache_key(contributing, train_json_sha256, cfg, fdr_threshold)
    genes = sorted({g for b in bdirs for g in b.genes})
    perts = sorted({p for b in bdirs for p in b.pert_names})
    gi, pi = ({g: i for i, g in enumerate(genes)}, {p: i for i, p in enumerate(perts)})
    n_p, n_g, n_k = (len(perts), len(genes), len(bdirs))
    rng = np.random.default_rng(cfg.seed)
    clip_samples = []
    for b in bdirs:
        ix = b.kept
        if len(ix) > 256:
            ix = np.sort(rng.choice(ix, 256, replace=False))
        fc = np.asarray(b.fc[ix], np.float32)
        valid = np.asarray(b.tested[ix], bool) & np.isfinite(fc) & (fc > 0)
        vals = np.abs(np.log2(fc[valid]))
        if len(vals) > 200000:
            vals = rng.choice(vals, 200000, replace=False)
        clip_samples.append(vals)
        release(b.fc)
        release(b.tested)
    vals = np.concatenate(clip_samples)
    limit = float(np.percentile(vals, cfg.lfc_clip_percentile)) if len(vals) else 1.0
    rules = ev.Rules(float(fdr_threshold), 1.0, max(limit, 1e-06))
    del clip_samples, vals, fc
    arrays = {
        k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(n_p, n_g))
        for k, dt in ev.GLOBAL_ARRAYS.items()
    }
    by_ds = {
        k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(n_k, n_p, n_g))
        for k, dt in ev.PER_DATASET_ARRAYS.items()
    }
    accumulated = {}
    chunk = min(cfg.chunk, 32)
    print(
        "EVIDENCE",
        key[:12],
        n_p,
        "perts",
        n_g,
        "genes",
        n_k,
        "datasets; disk-backed accumulation",
        flush=True,
    )
    for slot, b in enumerate(bdirs):
        columns = np.array([gi[g] for g in b.genes], np.int64)
        perturbation_ids = np.array([pi[p] for p in b.pert_names], np.int64)
        for start in range(0, len(b.kept), chunk):
            ix = b.kept[start : start + chunk]
            own = ev._own_row_stats(
                fc=b.fc[ix], fdr=b.fdr[ix], delta=b.dp[ix], tested=b.tested[ix], rules=rules
            )
            pp = perturbation_ids[start : start + chunk]
            index = (pp[:, None], columns[None, :])
            unique = len(set(pp.tolist())) == len(pp)
            for k in ev.GLOBAL_ARRAYS:
                if not own[k].any():
                    continue
                if unique:
                    arrays[k][index] += own[k].astype(arrays[k].dtype, copy=False)
                else:
                    np.add.at(arrays[k], index, own[k])
            for k, source in ev.PER_DATASET_SOURCE.items():
                if not own[source].any():
                    continue
                if unique:
                    by_ds[k][slot][index] += own[source].astype(by_ds[k].dtype, copy=False)
                else:
                    np.add.at(by_ds[k][slot], index, own[source])
            if start % 1024 == 0:
                for a in (*arrays.values(), *by_ds.values()):
                    release(a)
                for a in (b.fc, b.fdr, b.dp, b.tested):
                    release(a)
        accumulated[b.dataset] = len(b.kept)
        for a in (*arrays.values(), *by_ds.values()):
            release(a)
        for a in (b.fc, b.fdr, b.dp, b.tested):
            release(a)
        print("EVIDENCE ACCUMULATED", b.dataset, len(b.kept), flush=True)
    totals = {k: 0.0 for k in ("dp_sum", "dp_sumsq", "dp_cnt", "lfc_sum", "lfc_sumsq", "lfc_cnt")}
    for k in totals:
        for start in range(0, n_p, 256):
            totals[k] += float(arrays[k][start : start + 256].sum(dtype=np.float64))
        release(arrays[k])
    scales = ev._scales_from_totals(totals)
    section = ev._build_context_response(
        out, bdirs, rules=rules, train_json_sha256=train_json_sha256, chunk=chunk
    )
    meta = {
        "schema_version": 4,
        "quantity": ev.QUANTITY,
        "cache_key": key,
        "contributing_datasets": [b.dataset for b in bdirs],
        "train_json_sha256": train_json_sha256,
        "n_rows_accumulated": sum(accumulated.values()),
        "n_rows_per_dataset": accumulated,
        "rules": asdict(rules),
        "scales": scales.to_dict(),
        "lfc_clip_percentile": cfg.lfc_clip_percentile,
        "seed": cfg.seed,
        "chunk": cfg.chunk,
        "n_perts": n_p,
        "n_genes": n_g,
        "pert_to_index": pi,
        "gene_symbols": genes,
        "context_response": section,
        "builder": "bounded-memory float32/uint16/uint8 accumulation; validity bit for missing delta_p",
        "lfc_clip_fit": "training-only uniform sample, at most 256 rows per dataset",
    }
    ev.write_json(out / ev.RESPONSE_META, meta)
    for a in (*arrays.values(), *by_ds.values()):
        release(a)
    print("EVIDENCE COMPLETE", key[:12], flush=True)
    return out
