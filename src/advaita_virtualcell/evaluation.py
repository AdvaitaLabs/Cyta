"""Count-matched local diagnostics using the cell-eval raw metric functions."""

from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .io import canonical_names, read_json, resolve_path, validate_counts, write_json
from .preparation import de_table


def log_pseudobulk(counts):
    total = np.asarray(counts.sum(0)).ravel().astype(float)
    if total.sum() <= 0:
        raise ValueError("Pseudobulk requires a positive library")
    return np.log1p(5e4 * total / total.sum())


def evaluate_from_config(path):
    from cell_eval2.de import prepare_de, resolve_target_genes
    from cell_eval2.metrics.de import de_lfc_nmae
    from cell_eval2.metrics.direction import de_direction_reach
    from cell_eval2.metrics.discrimination import discrimination_score

    path = Path(path).resolve()
    recipe = read_json(path)
    root = path.parent
    prediction = ad.read_h5ad(resolve_path(recipe["predictions"], root))
    truth = ad.read_h5ad(resolve_path(recipe["truth"], root))
    genes = canonical_names(truth.var_names, "genes")
    if not np.array_equal(genes, canonical_names(prediction.var_names, "prediction genes")):
        raise ValueError("Prediction and truth gene axes must match")
    truth.X, prediction.X = validate_counts(truth.X), validate_counts(prediction.X)
    output = resolve_path(recipe["output"], root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target_key = recipe.get("target_column", "target_gene")
    context_key = recipe.get("context_column", "context")
    control = str(recipe.get("control_label", "non-targeting")).upper()
    seed = int(recipe.get("seed", 20261029))
    cap = int(recipe.get("cell_cap", 400))
    targets = pd.read_csv(resolve_path(recipe["targets"], root))["target_gene"].to_numpy(str)
    canonical_names(targets, "targets")
    rows = []
    for context in recipe.get("contexts", sorted(set(prediction.obs[context_key].astype(str)))):
        observed = truth[truth.obs[context_key].astype(str).to_numpy() == context]
        generated = prediction[prediction.obs[context_key].astype(str).to_numpy() == context]
        true_labels = observed.obs[target_key].astype(str).str.upper().to_numpy()
        pred_labels = generated.obs[target_key].astype(str).str.upper().to_numpy()
        reference = observed.X[true_labels == control]
        if reference.shape[0] < 2:
            raise ValueError(f"At least two reference controls are required for {context}")
        names, real, predicted = [], [], []
        for index, target in enumerate(targets):
            target = target.upper()
            actual = observed.X[true_labels == target][:cap]
            estimate = generated.X[pred_labels == target]
            if actual.shape[0] < 2:
                continue
            if estimate.shape[0] < actual.shape[0]:
                raise ValueError(
                    f"Insufficient predicted cells for {context}/{target}; cells are never duplicated"
                )
            chosen = np.random.default_rng(seed + index).choice(
                estimate.shape[0], actual.shape[0], replace=False
            )
            names.append(target)
            real.append(actual)
            predicted.append(estimate[chosen])
        if len(names) < 2:
            raise ValueError(f"PDS requires at least two eligible targets for {context}")
        real_de = de_table(reference, real, names, genes)
        pred_de = de_table(reference, predicted, names, genes)
        real_de.write_parquet(output / f"{context}_truth.parquet")
        pred_de.write_parquet(output / f"{context}_predicted.parquet")
        mapping = {target: target for target in names if target in set(genes)}
        resolution = resolve_target_genes(real_de, names, target_gene_map=mapping)
        prepared = prepare_de(pred_de, real_de, control=control, target_resolution=resolution)
        nmae = de_lfc_nmae(prepared)
        reach = de_direction_reach(prepared, universe="adjudicated", corrected=False)
        pred_bulk = np.stack([log_pseudobulk(matrix) for matrix in predicted])
        real_bulk = np.stack([log_pseudobulk(matrix) for matrix in [reference, *real]])
        pds = discrimination_score(
            pred_bulk=(np.asarray(names), pred_bulk),
            real_bulk=(np.asarray([control, *names]), real_bulk),
            genes=genes,
            control=control,
            control_source="real",
            distance="cosine",
            rank_denominator="n-1",
            tie_policy="midrank",
            exclusion_scope="panel",
        )
        for index, target in enumerate(names):
            mask = genes != target
            rows.append(
                dict(
                    context=context,
                    target=target,
                    n_cells=real[index].shape[0],
                    nmae=nmae.get(target, np.nan),
                    reach=reach.get(target, np.nan),
                    pds_cosine=pds[target],
                    raw_bulk_mse=float(
                        np.mean((pred_bulk[index, mask] - real_bulk[index + 1, mask]) ** 2)
                    ),
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    summary = {}
    for context, group in frame.groupby("context", sort=True):
        summary[context] = {
            key: dict(
                mean=float(group[key].mean()) if group[key].notna().any() else None,
                eligible_targets=int(group[key].notna().sum()),
            )
            for key in ("nmae", "reach", "pds_cosine", "raw_bulk_mse")
        }
    result = dict(
        by_context=summary,
        cell_eval_version=version("cell-eval2"),
        official_composite=False,
        count_matched=True,
        cell_cap=cap,
        seed=seed,
        bulk_axis="full gene axis with the target gene excluded; PDS excludes panel targets",
        mse="raw squared log1p(CPM50K) difference; not noise-corrected or normalized",
    )
    write_json(output / "summary.json", result)
    return result
