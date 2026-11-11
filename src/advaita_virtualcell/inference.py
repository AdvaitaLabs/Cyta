"""Export significance, direction, amplitude and pooled-response readouts."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from . import MODEL_NAME
from .backend import (
    autocast,
    configure_assets,
    construct_model,
    load_model_checkpoint,
    require_pie,
    select_device,
    subset_sample,
)
from .controls import control_statistics
from .evidence import install_evidence_extension
from .io import canonical_names, file_sha256, read_json, resolve_path, write_json


def infer_from_config(path):
    require_pie()
    import torch
    from pie.data.datamodule import PieDataModule
    from pie.data.dataset import collate
    from pie.data.preprocessed import PreprocessedMeta, write_preprocessed
    from pie.model import readout
    from pie.predict import _batch_to

    path = Path(path).resolve()
    recipe = read_json(path)
    root = path.parent
    data_overrides = configure_assets(resolve_path(recipe["assets"], root))
    install_evidence_extension()
    device = select_device(recipe.get("device", "auto"))
    precision = recipe.get("precision", "auto")
    torch.set_num_threads(int(recipe.get("threads", 2)))
    output = resolve_path(recipe["output"], root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {role: resolve_path(value, root) for role, value in recipe["checkpoints"].items()}
    if not {"base", "amplitude", "bulk"}.issubset(paths):
        raise ValueError("Checkpoints must define base, amplitude, and bulk roles")
    loaded = {role: load_model_checkpoint(value, data_overrides) for role, value in paths.items()}
    base = loaded["base"]
    text_keys = read_json(Path(base.config.data.gene_text_dir) / "meta.json")["keys"]
    canonical = {gene.upper(): gene for gene in text_keys}
    targets = pd.read_csv(resolve_path(recipe["targets"], root))[
        recipe.get("target_column", "target_gene")
    ].to_numpy(str)
    canonical_names(targets, "targets")
    manifest = {
        "model_name": MODEL_NAME,
        "contexts": {},
        "device": str(device),
        "precision": precision,
    }
    for context, value in recipe["controls"].items():
        if not context or any(char in context for char in "/\\"):
            raise ValueError("Context names must be nonempty path components")
        controls = ad.read_h5ad(resolve_path(value, root))
        full_genes = controls.var_names.to_numpy(str)
        canonical_names(full_genes, "genes")
        columns = np.array([i for i, gene in enumerate(full_genes) if gene.upper() in canonical])
        genes = [canonical[full_genes[i].upper()] for i in columns]
        if not len(genes):
            raise ValueError("No control genes match the gene-query embedding axis")
        statistics = control_statistics(controls.X)
        dataset = "advaita_query_" + context.lower()
        name = "anon_" + context.lower()
        query = output / "queries" / context
        query.mkdir(parents=True)
        meta = PreprocessedMeta(
            format_version=1,
            dataset=dataset,
            genes=genes,
            context_to_id={name: 0},
            pert_to_id={},
            pert_kind="gene",
            control_label="non-targeting",
            num_rows=0,
            num_genes=len(genes),
            num_contexts=1,
            num_perts=0,
            controls_only=True,
            tool_version=MODEL_NAME,
            array_sha256={},
        )
        write_preprocessed(
            query,
            meta,
            {"ctrl_means.npy": statistics["features"][columns][None].astype(np.float32)},
        )
        write_json(
            query / "query.json",
            {dataset + "." + name: [canonical.get(t.upper(), t) for t in targets]},
        )
        data = base.config.data.model_copy(
            update={"preprocessed_dirs": [str(query)], "dataset_weights": None, "num_workers": 0}
        )
        module = PieDataModule(
            data, None, base.stats, seed=base.config.seed, evidence_source=base.config.data
        )
        module.setup_stats(0)
        references = module.rows_from_query(query / "query.json")
        samples = module.make_dataset(references, "none")
        gene_lookup = {
            canonical[gene.upper()]: i
            for i, gene in enumerate(full_genes)
            if gene.upper() in canonical
        }
        axis = np.array([gene_lookup[gene] for gene in module.gene_axis])
        models = {
            role: construct_model(checkpoint, module, device) for role, checkpoint in loaded.items()
        }
        arrays = {role: {field: [] for field in ("p_de", "lfc", "delta_p")} for role in models}
        with torch.inference_mode(), autocast(device, precision):
            for index in range(len(samples)):
                sample = samples[index]
                sample = subset_sample(sample, np.arange(len(sample["gene_ids"])))
                batch = _batch_to(collate([sample], module.source_dims), device, None)
                for role, model in models.items():
                    checkpoint = loaded[role]
                    prediction = readout(
                        model(batch)[0], checkpoint.config.model, checkpoint.stats.delta_p
                    )
                    for field in arrays[role]:
                        values = getattr(prediction, field)[0].float().cpu().numpy()
                        if not np.isfinite(values).all():
                            raise FloatingPointError(f"Nonfinite {role}/{field} prediction")
                        arrays[role][field].append(values)
                if (index + 1) % 10 == 0:
                    print(f"Inferred {context}: {index + 1}/{len(samples)} targets", flush=True)
        manifest["contexts"][context] = {}
        for role in models:
            destination = output / role / f"{context}.npz"
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                destination,
                genes=np.asarray(module.gene_axis),
                targets=np.array([r.perturbation for r in references]),
                control_mean_cpm=statistics["arithmetic"][axis] * 1e6,
                control_pool_fraction=statistics["pooled"][axis],
                **{key: np.stack(values) for key, values in arrays[role].items()},
            )
            manifest["contexts"][context][role] = {
                "path": str(destination.relative_to(output)),
                "checkpoint_sha256": file_sha256(paths[role]),
                "prediction_sha256": file_sha256(destination),
            }
        del models, arrays, module, samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
