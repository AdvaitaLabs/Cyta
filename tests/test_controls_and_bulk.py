import numpy as np
import pytest
from scipy import sparse

from advaita_virtualcell.bulk import correct_common_response, pooled_profiles
from advaita_virtualcell.config import RenderConfig
from advaita_virtualcell.controls import (
    control_statistics,
    render_control_statistics,
    select_control_cells,
)
from advaita_virtualcell.io import align_readouts, validate_counts
from advaita_virtualcell.renderer import render_target
from advaita_virtualcell.routing import BackgroundRouter


def test_arithmetic_and_pooled_means_are_different_estimands():
    matrix = sparse.csr_matrix([[9, 1], [10, 90]], dtype=np.int32)
    stats = control_statistics(matrix)
    np.testing.assert_allclose(stats["arithmetic"], [0.5, 0.5])
    np.testing.assert_allclose(stats["pooled"], [19 / 110, 91 / 110])


def test_renderer_preserves_native_float32_mass_rounding():
    counts = np.full((400, 2), 30000, np.float32)
    counts[0, 0] += 1
    base = sparse.csr_matrix(counts)
    total = int(base.sum())
    assert total != int(base.astype(np.int64).sum())
    profile = np.array([0.5, 0.5])
    result, _ = render_target(
        base,
        profile,
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.ones(2, bool),
        -1,
        profile,
        profile,
        seed_index=0,
        config=RenderConfig(top_de_genes=0, repair_passes=0),
    )
    assert int(result.sum()) == total
    assert (np.asarray(result.sum(1)).ravel() > 0).all()


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.float64])
def test_streamed_render_statistics_match_original_reduction(dtype):
    rng = np.random.default_rng(91)
    matrix = sparse.csr_matrix(rng.poisson(3, (69, 27)).astype(dtype))
    matrix.data[matrix.data < 2] = 0
    matrix.eliminate_zeros()
    libraries = np.asarray(matrix.sum(1)).ravel()
    expected_mean = np.asarray((sparse.diags(1 / libraries) @ matrix).mean(0)).ravel()
    expected_pool = np.asarray(matrix.sum(0)).ravel().astype(float)
    expected_pool /= expected_pool.sum()
    stats = render_control_statistics(matrix, chunk_size=11)
    np.testing.assert_array_equal(stats["arithmetic"], expected_mean)
    np.testing.assert_array_equal(stats["pooled"], expected_pool)


def test_control_selection_is_reproducible_and_unique_when_possible():
    source = np.random.default_rng(12).poisson(3, (45, 16))
    first = select_control_cells(source, 5, 12)
    np.testing.assert_array_equal(first, select_control_cells(source, 5, 12))
    assert all(len(set(row)) == 12 for row in first)
    np.testing.assert_array_equal(
        select_control_cells(source[:12], 2, 12), np.tile(np.arange(12), (2, 1))
    )
    assert select_control_cells(source[:3], 2, 12).shape == (2, 12)


def test_router_uses_only_control_profiles_and_checks_axes():
    genes = np.array(["A", "B", "C"])
    profiles = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    router = BackgroundRouter(genes, np.array(["near", "fallback"]), profiles, "near")
    assert router.route(profiles[0], genes)["use_specialist"]
    assert not router.route(profiles[1], genes)["use_specialist"]
    with pytest.raises(ValueError, match="gene order"):
        router.route(profiles[0], genes[::-1])


def test_response_correction_keeps_probabilities_valid_and_preserves_linear_contrasts():
    genes = np.array(["A", "B", "C", "D"])
    targets = np.array(["X", "Y"])
    control = np.array([0.2, 0.3, 0.1, 0.4])
    lfc = np.array([[0.1, -0.2, 0.3, 0.1], [-0.1, 0.3, -0.1, -0.1]])
    profiles = pooled_profiles(control, lfc, genes, targets)
    basis = np.eye(4)[:, :2]
    corrected, shift = correct_common_response(
        profiles, control, genes, targets, np.ones(4, bool), basis
    )
    delta = np.log1p(5e4 * profiles) - np.log1p(5e4 * control)
    np.testing.assert_allclose(
        (delta + shift)[0] - (delta + shift)[1], delta[0] - delta[1], atol=1e-15
    )
    np.testing.assert_allclose(corrected.sum(1), 1)
    assert (corrected >= 0).all() and np.isfinite(corrected).all()


def test_readout_alignment_keeps_missing_genes_explicit():
    archive = {
        "genes": np.array(["B", "A"]),
        "targets": np.array(["T"]),
        "lfc": np.array([[2.0, 1.0]]),
    }
    values, available = align_readouts(archive, ["a", "C", "b"], ["t"])
    np.testing.assert_array_equal(values, [[1, 0, 2]])
    np.testing.assert_array_equal(available, [True, False, True])
    with pytest.raises(ValueError, match="missing requested targets"):
        align_readouts(archive, ["A"], ["UNKNOWN"])
    with pytest.raises(ValueError, match="Raw integer"):
        validate_counts([[0.5, 1]])
