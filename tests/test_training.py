from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from advaita_virtualcell.training.foundation import RowCycle, average_precision
from advaita_virtualcell.training.losses import (
    bulk_components,
    contrastive_bulk_loss,
    specialist_loss,
)
from advaita_virtualcell.training.runner import set_trainable
from advaita_virtualcell.training.selection import bulk_candidate, specialist_candidate


def test_specialist_loss_masks_missing_labels_and_keeps_finite_gradients():
    raw = torch.full((16,), 0.4, requires_grad=True)
    direction = torch.ones(16)
    truth = torch.full((16,), 0.6)
    truth[-1] = float("nan")
    tested = torch.ones(16, dtype=torch.bool)
    significant = tested.clone()
    loss = specialist_loss(
        raw, direction, torch.full((16,), 0.3), truth, tested, significant, torch.ones(16)
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert (raw.grad[:-1] < 0).all()


def test_contrastive_bulk_loss_rewards_correct_target_pairing():
    control = torch.full((8,), 1 / 8)
    responses = torch.tensor([[0.6, -0.6] * 4, [-0.6, 0.6] * 4])
    truths = control * torch.exp2(responses)
    truths /= truths.sum(1, keepdim=True)
    available = torch.ones(8, dtype=torch.bool)
    correct = [bulk_components(responses[i], control, truths[i], available) for i in range(2)]
    swapped = [bulk_components(responses[i], control, truths[1 - i], available) for i in range(2)]
    assert contrastive_bulk_loss(correct) < contrastive_bulk_loss(swapped)
    raw = responses.clone().requires_grad_()
    loss = contrastive_bulk_loss(
        [bulk_components(raw[i], control, truths[i], available) for i in range(2)]
    )
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_readout_adaptation_freezes_backbone_and_other_heads():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(3, 3)
    model.lfc_head = torch.nn.Linear(3, 1)
    model.output_head = torch.nn.Linear(3, 2)
    model.evidence_output_adapters = torch.nn.ModuleDict(
        {"lfc": torch.nn.Linear(3, 1), "de": torch.nn.Linear(3, 2)}
    )
    set_trainable(model, "specialist")
    assert model.lfc_head.weight.requires_grad
    assert model.evidence_output_adapters["lfc"].weight.requires_grad
    assert not model.backbone.weight.requires_grad
    assert not model.output_head.weight.requires_grad
    assert not model.evidence_output_adapters["de"].weight.requires_grad


def test_selection_requires_both_near_gain_and_source_guard():
    assert specialist_candidate(1.001, 0.99, 1, 1)["passed"]
    assert not specialist_candidate(1.003, 0.8, 1, 1)["passed"]
    assert not specialist_candidate(0.99, 0.999, 1, 1)["passed"]
    assert bulk_candidate(0.01, 0.70, 0.60, 0.70, 0.60)
    assert not bulk_candidate(0.001, 0.68, 0.60, 0.70, 0.60)


def test_row_cycle_covers_each_context_before_repeating():
    rows = [SimpleNamespace(dir_index=0, context="k562_gwps") for _ in range(7)]
    dataset = SimpleNamespace(rows=rows, dirs=[SimpleNamespace(dataset="replogle")])
    cycle = RowCycle(dataset, np.random.default_rng(4), 0.1)
    positions = cycle.batch(7)
    assert len(set(positions)) == 7
    assert len(cycle.seen) == 7
    assert average_precision(
        np.array([True, False, True]), np.array([0.9, 0.8, 0.7])
    ) == pytest.approx(5 / 6)
