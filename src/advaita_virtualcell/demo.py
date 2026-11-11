"""Small deterministic synthetic inputs for an offline pipeline smoke test."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .assembly import assemble
from .io import write_json
from .pipeline import render_from_config
from .routing import BackgroundRouter
from .validation import validate_h5ad


def run_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Demo directory must be empty: {output}")
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    genes = np.asarray([f"GENE_{index:03d}" for index in range(48)])
    targets = genes[:4]
    counts = rng.poisson(rng.uniform(2, 8, len(genes)), (40, len(genes))).astype(np.int32)
    controls = ad.AnnData(
        sparse.csr_matrix(counts),
        var=pd.DataFrame(index=genes),
        obs=pd.DataFrame(index=[f"control_{index}" for index in range(len(counts))]),
    )
    controls.write_h5ad(inputs / "controls.h5ad")
    pd.DataFrame({"target_gene": targets}).to_csv(inputs / "targets.csv", index=False)
    lfc = rng.normal(0, 0.25, (len(targets), len(genes))).astype(np.float32)
    for role, scale in (("base", 1.0), ("amplitude", 0.7), ("bulk", 0.4), ("specialist", 0.8)):
        np.savez_compressed(
            inputs / f"{role}.npz",
            genes=genes,
            targets=targets,
            lfc=lfc * scale,
            p_de=rng.uniform(0, 1, lfc.shape).astype(np.float32),
        )
    basis, _ = np.linalg.qr(rng.normal(size=(len(genes), 6)))
    np.savez_compressed(
        inputs / "basis.npz", genes=genes, basis=basis.astype(np.float32), alpha=0.5
    )
    pooled = counts.sum(0).astype(float)
    pooled /= pooled.sum()
    BackgroundRouter(
        genes,
        np.asarray(["specialist", "fallback"]),
        np.stack([pooled, pooled[::-1]]),
        "specialist",
    ).save(inputs / "router.npz")
    recipe = dict(
        targets="inputs/targets.csv",
        response_basis="inputs/basis.npz",
        router="inputs/router.npz",
        output="blocks",
        render=dict(cells_per_target=24, top_de_genes=12),
        contexts={
            "example": dict(
                controls="inputs/controls.h5ad",
                base_readouts="inputs/base.npz",
                amplitude_readouts="inputs/amplitude.npz",
                bulk_readouts="inputs/bulk.npz",
                specialist_readouts="inputs/specialist.npz",
            )
        },
    )
    write_json(output / "render.json", recipe)
    manifest = render_from_config(output / "render.json")
    assemble(manifest, output / "predictions.h5ad")
    return validate_h5ad(
        output / "predictions.h5ad",
        targets=inputs / "targets.csv",
        contexts=["example"],
        cells_per_target=24,
    )
