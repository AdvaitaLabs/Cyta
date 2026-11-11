"""Inference orchestration from portable response archives to count blocks."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from . import MODEL_NAME
from .bulk import correct_common_response, load_basis, pooled_profiles
from .config import RenderConfig
from .controls import CountMatrixReader, render_control_statistics, select_control_cells
from .io import (
    align_readouts,
    canonical_names,
    count_sha256,
    file_sha256,
    load_readouts,
    read_json,
    resolve_path,
    write_json,
)
from .renderer import render_target
from .routing import BackgroundRouter


def _render_job(job):
    with threadpool_limits(limits=1):
        output = Path(job.pop("output"))
        target = job.pop("target")
        context = job.pop("context")
        index = job["seed_index"]
        counts, info = render_target(**job)
        counts.eliminate_zeros()
        counts.sort_indices()
        temporary = output.with_suffix(".partial.npz")
        sparse.save_npz(temporary, counts)
        temporary.replace(output)
        record = {
            "context": context,
            "target": target,
            "index": index,
            "path": output.name,
            "shape": list(counts.shape),
            "nnz": int(counts.nnz),
            "counts_sha256": count_sha256(counts),
            **info,
        }
        write_json(output.with_suffix(".json"), record)
        return record


def render_from_config(config_path, *, indices=None, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    config_path = Path(config_path).resolve()
    recipe = read_json(config_path)
    root = config_path.parent
    settings = RenderConfig(**recipe.get("render", {}))
    targets_frame = pd.read_csv(resolve_path(recipe["targets"], root))
    targets = targets_frame[recipe.get("target_column", "target_gene")].to_numpy(str)
    canonical_names(targets, "targets")
    chosen = list(range(len(targets))) if indices is None else list(indices)
    if (
        not chosen
        or len(set(chosen)) != len(chosen)
        or min(chosen) < 0
        or max(chosen) >= len(targets)
    ):
        raise ValueError("Target indices must be distinct and within the complete target list")
    output = resolve_path(recipe["output"], root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    router = (
        BackgroundRouter.load(resolve_path(recipe["router"], root))
        if recipe.get("router")
        else None
    )
    records = []
    metadata = {}
    expected_genes = None
    for context, inputs in recipe["contexts"].items():
        if not context or any(c in context for c in "/\\"):
            raise ValueError("Context names must be nonempty path components")
        controls_path = resolve_path(inputs["controls"], root)
        controls = ad.read_h5ad(controls_path, backed="r")
        counts = CountMatrixReader(controls_path)
        genes = controls.var_names.to_numpy(str)
        canonical_names(genes, "genes")
        if expected_genes is None:
            expected_genes = genes
            pd.DataFrame({"gene_name": genes}).to_csv(output / "genes.csv", index=False)
        elif not np.array_equal(expected_genes, genes):
            raise ValueError("All contexts must use the same full output gene order")
        statistics = render_control_statistics(counts)
        route = (
            router.route(statistics["projection_pooled"], genes)
            if router
            else {"nearest": None, "use_specialist": False, "scores": {}}
        )
        amplitude_key = "specialist_readouts" if route["use_specialist"] else "amplitude_readouts"
        if amplitude_key not in inputs:
            raise ValueError(f"Missing {amplitude_key} for context {context}")
        archives = {
            key: load_readouts(resolve_path(inputs[key], root))
            for key in ("base_readouts", "bulk_readouts", amplitude_key)
        }
        if "p_de" not in archives["base_readouts"]:
            raise ValueError("Base readouts must include p_de")
        base_lfc, available = align_readouts(archives["base_readouts"], genes, targets)
        probabilities, _ = align_readouts(archives["base_readouts"], genes, targets, "p_de")
        amplitude, amp_available = align_readouts(archives[amplitude_key], genes, targets)
        bulk_lfc, bulk_available = align_readouts(archives["bulk_readouts"], genes, targets)
        if not np.array_equal(available, amp_available) or not np.array_equal(
            available, bulk_available
        ):
            raise ValueError("All readout branches must cover the same modeled genes")
        baseline = pooled_profiles(
            statistics["pooled"],
            bulk_lfc,
            genes,
            targets,
            cap=settings.amplitude_cap,
            target_retention=settings.target_retention,
        )
        projection_input = pooled_profiles(
            statistics["projection_pooled"],
            bulk_lfc.astype(np.float32),
            genes,
            targets,
            cap=settings.amplitude_cap,
            target_retention=settings.target_retention,
        )
        basis, alpha = load_basis(resolve_path(recipe["response_basis"], root), genes)
        corrected, shift = correct_common_response(
            projection_input,
            statistics["projection_pooled"],
            genes,
            targets,
            available,
            basis,
            alpha=alpha,
            cap=settings.amplitude_cap,
            target_retention=settings.target_retention,
        )
        readout_hashes = {key: file_sha256(resolve_path(inputs[key], root)) for key in archives}
        del archives, bulk_lfc, projection_input, basis
        if inputs.get("control_sets"):
            sets = np.load(resolve_path(inputs["control_sets"], root), allow_pickle=False)
            if not np.issubdtype(sets.dtype, np.integer):
                raise ValueError("Control selections must contain integer row indices")
        else:
            with threadpool_limits(limits=1):
                sets = select_control_cells(counts[:], len(targets), settings.cells_per_target)
        if (
            sets.shape != (len(targets), settings.cells_per_target)
            or np.any(sets < 0)
            or np.any(sets >= counts.shape[0])
        ):
            raise ValueError("Invalid representative control selections")
        if counts.shape[0] >= settings.cells_per_target and any(
            len(set(row)) != len(row) for row in sets
        ):
            raise ValueError("Representative selections contain unnecessary duplicate cells")
        context_dir = output / context
        context_dir.mkdir()
        np.save(context_dir / "control_sets.npy", sets)
        np.savez_compressed(
            context_dir / "bulk_profiles.npz",
            genes=genes,
            targets=targets,
            baseline=baseline,
            corrected=corrected,
            common_shift=shift,
        )
        gene_lookup = dict(zip(canonical_names(genes), range(len(genes))))

        def jobs():
            for index in chosen:
                yield {
                    "output": str(context_dir / f"{index:04d}.npz"),
                    "context": context,
                    "target": str(targets[index]),
                    "base": counts[sets[index]],
                    "control_mean": statistics["arithmetic"],
                    "base_lfc": base_lfc[index],
                    "amplitude_lfc": amplitude[index],
                    "probabilities": probabilities[index],
                    "available": available,
                    "target_index": gene_lookup.get(str(targets[index]).upper(), -1),
                    "baseline_profile": baseline[index],
                    "corrected_profile": corrected[index],
                    "seed_index": index,
                    "config": settings,
                    "specialist": route["use_specialist"],
                }

        if workers == 1:
            context_records = list(map(_render_job, jobs()))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                remaining = iter(jobs())
                pending = set()
                context_records = []
                for _ in range(2 * workers):
                    job = next(remaining, None)
                    if job is not None:
                        pending.add(pool.submit(_render_job, job))
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        context_records.append(future.result())
                        job = next(remaining, None)
                        if job is not None:
                            pending.add(pool.submit(_render_job, job))
                positions = {index: position for position, index in enumerate(chosen)}
                context_records.sort(key=lambda record: positions[record["index"]])
        records.extend(context_records)
        metadata[context] = {
            "route": route,
            "n_controls": int(counts.shape[0]),
            "controls_sha256": file_sha256(controls_path),
            "readout_sha256": readout_hashes,
        }
        controls.file.close()
        counts.close()
        print(f"Rendered {context}: {len(context_records)} targets", flush=True)
    manifest = {
        "model_name": MODEL_NAME,
        "complete": True,
        "settings": asdict(settings),
        "targets": targets.tolist(),
        "selected_indices": chosen,
        "genes": expected_genes.tolist(),
        "contexts": metadata,
        "blocks": records,
        "common_response_uses_complete_target_list": True,
    }
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
