import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from advaita_virtualcell.assembly import assemble
from advaita_virtualcell.demo import run_demo
from advaita_virtualcell.io import read_json, write_json
from advaita_virtualcell.pipeline import render_from_config
from advaita_virtualcell.validation import validate_h5ad


@pytest.fixture
def demo(tmp_path):
    root = tmp_path / "demo"
    result = run_demo(root)
    assert result["valid"] and result["shape"] == [96, 48]
    return root


def test_demo_covers_all_targets_with_integer_counts(demo):
    data = ad.read_h5ad(demo / "predictions.h5ad")
    assert data.X.dtype == np.int32
    assert data.uns["model_name"] == "Advaita_VirtualCell_Model"
    assert data.obs.groupby("target_gene", observed=True).size().tolist() == [24] * 4
    manifest = read_json(demo / "blocks/manifest.json")
    assert all(row["transport"]["minimal_count_movement"] for row in manifest["blocks"])
    assert manifest["contexts"]["example"]["route"]["use_specialist"]


def test_partial_render_uses_full_target_design_and_original_seeds(demo):
    recipe = read_json(demo / "render.json")
    recipe["output"] = "partial"
    write_json(demo / "partial.json", recipe)
    render_from_config(demo / "partial.json", indices=[2])
    expected = sparse.load_npz(demo / "blocks/example/0002.npz")
    actual = sparse.load_npz(demo / "partial/example/0002.npz")
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())


def test_parallel_render_matches_serial_counts_and_order(demo):
    recipe = read_json(demo / "render.json")
    recipe["output"] = "parallel"
    write_json(demo / "parallel.json", recipe)
    render_from_config(demo / "parallel.json", workers=2)
    serial = read_json(demo / "blocks/manifest.json")["blocks"]
    parallel = read_json(demo / "parallel/manifest.json")["blocks"]
    assert [(row["index"], row["counts_sha256"]) for row in serial] == [
        (row["index"], row["counts_sha256"]) for row in parallel
    ]


def test_assembly_rejects_tampered_counts(demo):
    block = demo / "blocks/example/0000.npz"
    matrix = sparse.load_npz(block)
    matrix.data[0] += 1
    sparse.save_npz(block, matrix)
    with pytest.raises(ValueError, match="integrity"):
        assemble(demo / "blocks/manifest.json", demo / "tampered.h5ad")


def test_assembly_and_validation_reject_missing_targets(demo):
    path = demo / "blocks/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["blocks"].pop()
    write_json(demo / "blocks/incomplete.json", manifest)
    with pytest.raises(ValueError, match="missing expected"):
        assemble(demo / "blocks/incomplete.json", demo / "incomplete.h5ad")
    with pytest.raises(ValueError, match="contexts"):
        validate_h5ad(demo / "predictions.h5ad", contexts=["wrong"], cells_per_target=24)
