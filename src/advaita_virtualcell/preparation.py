"""Prepare source response archives, specialist labels and control references."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .controls import control_statistics
from .io import canonical_names, read_json, resolve_path, validate_counts, write_json
from .routing import BackgroundRouter


def _read_counts(path, genes=None):
    data = ad.read_h5ad(path)
    canonical_names(data.var_names, "genes")
    if genes is not None and not np.array_equal(
        canonical_names(data.var_names), canonical_names(genes)
    ):
        raise ValueError("Input files must have the same gene order")
    data.X = validate_counts(data.X).tocsr()
    return data


def _destination(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prepare_source(path):
    path = Path(path).resolve()
    recipe = read_json(path)
    output = resolve_path(recipe["output"], path.parent)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for context, inputs in recipe["contexts"].items():
        if not context or any(char in context for char in "/\\"):
            raise ValueError("Contexts must be nonempty path components")
        destination = _destination(output / f"{context}.npz")
        data = _read_counts(resolve_path(inputs["counts"], path.parent))
        labels = data.obs[inputs.get("target_column", "target_gene")].astype(str).to_numpy()
        control_label = inputs.get("control_label", "non-targeting")
        control = data.X[labels == control_label]
        statistics = control_statistics(control)
        requested = inputs.get("targets", sorted(set(labels) - {control_label}))
        targets, pools, arithmetic, counts = [], [], [], []
        for target in requested:
            cells = data.X[labels == target]
            if cells.shape[0] < inputs.get("min_cells", 100):
                continue
            values = control_statistics(cells)
            targets.append(target)
            pools.append(values["pooled"])
            arithmetic.append(values["arithmetic"])
            counts.append(cells.shape[0])
        canonical_names(targets, "targets")
        shape = (len(targets), data.n_vars)
        np.savez_compressed(
            destination,
            genes=data.var_names.to_numpy(str),
            targets=np.asarray(targets),
            pool=np.asarray(pools, np.float32),
            mean=np.asarray(arithmetic, np.float32),
            control=np.broadcast_to(statistics["pooled"], shape).astype(np.float32),
            features=np.broadcast_to(statistics["features"], shape).astype(np.float32),
            counts=np.asarray(counts, np.int64),
        )
        summary[context] = dict(targets=len(targets), controls=int(control.shape[0]))
    write_json(output / "manifest.json", summary)
    return summary


def cpm_moments(counts):
    counts = validate_counts(counts).astype(np.float64)
    scaled = sparse.diags(1e6 / np.asarray(counts.sum(1)).ravel()) @ counts
    mean = np.asarray(scaled.mean(0)).ravel()
    variance = np.maximum(np.asarray(scaled.multiply(scaled).mean(0)).ravel() - mean**2, 0)
    variance *= counts.shape[0] / max(counts.shape[0] - 1, 1)
    return mean, variance


def precision_weights(target, reference, width=0.25):
    if target.shape[0] < 2 or reference.shape[0] < 2 or width <= 0:
        raise ValueError(
            "Precision estimation requires at least two cells per group and positive width"
        )
    tm, tv = cpm_moments(target)
    cm, cv = cpm_moments(reference)
    variance = (
        tv / target.shape[0] / np.maximum(tm, 1e-6) ** 2
        + cv / reference.shape[0] / np.maximum(cm, 1e-6) ** 2
    ) / np.log(2) ** 2
    return 1 / (1 + variance / width**2)


def de_table(reference, groups, targets, genes):
    try:
        from cell_eval2.de_compute import compute_de_streaming_cell
    except ImportError as error:
        raise ImportError("Differential expression requires the 'evaluation' extra") from error
    return compute_de_streaming_cell(
        ref_X=reference,
        group_iter_factory=lambda: iter(enumerate(groups)),
        targets=np.asarray(targets, str),
        var_names=canonical_names(genes),
        n_genes=len(genes),
        backend="gpudge",
        mean_calc="arithmetic",
        epsilon=1e-9,
        target_sum=1e6,
        clip_value=None,
        fdr_scope="per_pert",
        filter_gene_min_cpm_cell=5.0,
    )


def prepare_specialist(path):
    path = Path(path).resolve()
    recipe = read_json(path)
    root = path.parent
    output = _destination(resolve_path(recipe["output"], root))
    adaptation = _read_counts(resolve_path(recipe["adaptation"], root))
    genes = adaptation.var_names.to_numpy(str)
    reference = _read_counts(resolve_path(recipe["reference_controls"], root), genes)
    query = _read_counts(resolve_path(recipe["query_controls"], root), genes)
    for left, right in ((adaptation, reference), (adaptation, query), (reference, query)):
        if set(left.obs_names) & set(right.obs_names):
            raise ValueError(
                "Adaptation, query and reference cells must have distinct global cell identifiers"
            )
    frame = pd.read_csv(resolve_path(recipe["target_split"], root))
    targets = frame["target_gene"].to_numpy(str)
    canonical_names(targets, "targets")
    split = frame["split"].to_numpy(str)
    if not set(split) <= {"train", "validation"} or len(set(split)) != 2:
        raise ValueError("Target split must contain train and validation rows")
    labels = adaptation.obs[recipe.get("target_column", "target_gene")].astype(str).to_numpy()
    groups = [adaptation.X[labels == target] for target in targets]
    if any(group.shape[0] < 2 for group in groups):
        raise ValueError("Every specialist target needs at least two adaptation cells")
    table = de_table(reference.X, groups, targets, genes).to_pandas()
    table.to_parquet(output.with_suffix(".parquet"))
    lookup = dict(zip(canonical_names(genes), range(len(genes))))
    shape = (len(targets), len(genes))
    lfc = np.zeros(shape, np.float32)
    significant, tested = np.zeros(shape, bool), np.zeros(shape, bool)
    precision = np.zeros(shape, np.float32)
    for index, (target, cells) in enumerate(zip(targets, groups, strict=True)):
        values = table[table.target == target]
        columns = np.array([lookup[str(gene).upper()] for gene in values.feature])
        lfc[index, columns] = values.log2_fold_change
        significant[index, columns] = values.p_adj < 0.05
        tested[index, columns] = (values.ref_mean >= 5) & np.isfinite(values.log2_fold_change)
        own = lookup.get(target.upper(), -1)
        if own >= 0:
            significant[index, own] = tested[index, own] = False
        precision[index] = precision_weights(cells, reference.X)
        precision[index, ~tested[index]] = 0
    np.savez_compressed(
        output,
        genes=genes,
        targets=targets,
        lfc=lfc,
        significant=significant,
        tested=tested,
        precision=precision,
        is_validation=split == "validation",
        features=control_statistics(query.X)["features"].astype(np.float32),
    )
    result = dict(
        targets=len(targets),
        training=int((split == "train").sum()),
        validation=int((split == "validation").sum()),
        independent_reference=True,
        eligible_genes=(significant & tested).sum(1).tolist(),
    )
    write_json(output.with_suffix(".json"), result)
    return result


def fit_router(path):
    path = Path(path).resolve()
    recipe = read_json(path)
    profiles, names, genes = [], [], None
    for name, value in recipe["references"].items():
        source = resolve_path(value, path.parent)
        if source.suffix == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                axis = archive["genes"].astype(str)
                profile = archive["control"]
                profile = profile.mean(0) if profile.ndim == 2 else profile
        else:
            data = _read_counts(source)
            axis = data.var_names.to_numpy(str)
            profile = control_statistics(data.X)["pooled"]
        if genes is None:
            genes = axis
        elif not np.array_equal(canonical_names(genes), canonical_names(axis)):
            raise ValueError("All router references must use the same gene order")
        profiles.append(profile)
        names.append(name)
    router = BackgroundRouter(
        genes,
        np.asarray(names),
        np.stack(profiles),
        recipe["specialist"],
        float(recipe.get("expression_gate", 1e-5)),
    )
    router.save(_destination(resolve_path(recipe["output"], path.parent)))
    return dict(references=names, specialist=router.specialist, genes=len(genes))


def fit_basis(path):
    import torch

    from .backend import select_device
    from .evidence import lineage
    from .training.data import held_out_target

    path = Path(path).resolve()
    recipe = read_json(path)
    directory = resolve_path(recipe["source_bulk"], path.parent)
    rows, groups, genes = [], [], None
    for source in sorted(directory.glob("*.npz")):
        if lineage(source.stem) in recipe.get("exclude_lineages", []):
            continue
        with np.load(source, allow_pickle=False) as archive:
            axis = archive["genes"].astype(str)
            if genes is None:
                genes = axis
            elif not np.array_equal(canonical_names(genes), canonical_names(axis)):
                raise ValueError("Source responses must use a common gene order")
            lookup = dict(zip(canonical_names(genes), range(len(genes))))
            for index, target in enumerate(archive["targets"].astype(str)):
                if held_out_target(target):
                    continue
                delta = np.log1p(5e4 * archive["pool"][index]) - np.log1p(
                    5e4 * archive["control"][index]
                )
                own = lookup.get(target.upper(), -1)
                if own >= 0:
                    delta[own] = 0
                rows.append(delta)
                groups.append(lineage(source.stem))
    if not rows:
        raise ValueError("No source training responses remain after exclusions")
    counts = {group: groups.count(group) for group in set(groups)}
    weights = np.asarray([1 / np.sqrt(counts[group]) for group in groups], np.float32)
    matrix = torch.from_numpy(np.stack(rows) * weights[:, None]).to(
        select_device(recipe.get("device", "auto"))
    )
    rank, alpha = int(recipe.get("rank", 64)), float(recipe.get("alpha", 0.5))
    q = min(128, min(matrix.shape) - 1)
    if rank < 1 or rank > q or not 0 <= alpha <= 1:
        raise ValueError(f"Basis rank must be between 1 and {q}; alpha must lie in [0, 1]")
    torch.manual_seed(int(recipe.get("seed", 20261030)))
    with torch.no_grad():
        _, singular, basis = torch.pca_lowrank(matrix, q=q, center=False, niter=4)
    output = _destination(resolve_path(recipe["output"], path.parent))
    np.savez_compressed(
        output,
        genes=genes,
        basis=basis[:, :rank].cpu().numpy(),
        singular_values=singular[:rank].cpu().numpy(),
        alpha=alpha,
    )
    result = dict(
        rank=rank,
        alpha=alpha,
        source_responses=len(rows),
        source_lineages=counts,
        held_out_targets_excluded=True,
        device=str(matrix.device),
    )
    write_json(output.with_suffix(".json"), result)
    return result
