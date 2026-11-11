"""Objectives that distinguish deployed amplitude from pooled response."""

import torch
from torch.nn import functional as F


def deployed_amplitude(raw_amplitude, base_lfc, cap=1.5):
    return torch.where(base_lfc >= 0, 1.0, -1.0) * raw_amplitude.float().abs().clamp(max=cap)


def significant_nmae(prediction, truth, significant, *, training=False, min_genes=10):
    mask = significant & torch.isfinite(truth)
    if int(mask.sum()) < min_genes:
        return None
    labels = truth[mask].clamp(-4, 4) if training else truth[mask]
    floor = 0.1 if training else 0.01
    return (prediction[mask] - labels).abs().mean() / truth[mask].abs().mean().clamp_min(floor)


def specialist_loss(
    raw_amplitude,
    base_lfc,
    teacher_lfc,
    truth,
    tested,
    significant,
    precision,
    *,
    cap=1.5,
    huber_beta=0.25,
    significant_weight=0.25,
    nonsignificant_weight=0.10,
    teacher_weight=0.02,
):
    prediction = deployed_amplitude(raw_amplitude, base_lfc, cap)
    valid = tested & torch.isfinite(truth)
    significant = significant & valid
    if not torch.isfinite(precision).all() or (precision < 0).any():
        raise ValueError("Precision weights must be finite and nonnegative")
    loss = raw_amplitude.sum() * 0
    nmae = significant_nmae(prediction, truth, significant, training=True)
    if nmae is not None:
        loss = loss + nmae
    safe_truth = torch.where(valid, truth, torch.zeros_like(truth)).clamp(-4, 4)
    huber = F.smooth_l1_loss(prediction, safe_truth, beta=huber_beta, reduction="none")
    for mask, weight in (
        (significant, significant_weight),
        (valid & ~significant, nonsignificant_weight),
    ):
        if mask.any():
            loss = loss + weight * (huber[mask] * precision[mask]).sum() / precision[
                mask
            ].sum().clamp_min(1e-8)
    return loss + teacher_weight * (raw_amplitude.float() - teacher_lfc.float()).square().mean()


def bulk_components(lfc, control, truth, available, target_index=-1):
    mask = available & (control > 0)
    if target_index >= 0:
        mask = mask.clone()
        mask[target_index] = False
    if not mask.any():
        raise ValueError("No eligible bulk-response genes")
    prediction = control * torch.exp2(lfc.float().clamp(-1.5, 1.5))
    prediction = prediction * (control.sum() / prediction.sum().clamp_min(1e-12))
    predicted_delta = torch.log1p(5e4 * prediction) - torch.log1p(5e4 * control)
    true_delta = torch.log1p(5e4 * truth) - torch.log1p(5e4 * control)
    mse = (predicted_delta[mask] - true_delta[mask]).square().mean()
    cosine = F.cosine_similarity(predicted_delta[mask], true_delta[mask], dim=0, eps=1e-8)
    return mse, cosine, predicted_delta, true_delta, mask


def contrastive_bulk_loss(
    components, cosine_weight=0.10, contrastive_weight=0.02, temperature=0.15
):
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    common = torch.stack([row[4] for row in components]).all(0)
    if not common.any():
        raise ValueError("Contrastive pairs have no shared eligible genes")
    prediction = torch.stack([row[2] for row in components])[:, common]
    truth = torch.stack([row[3] for row in components])[:, common]
    similarities = F.normalize(prediction, dim=-1) @ F.normalize(truth, dim=-1).T
    contrastive = F.cross_entropy(
        similarities / temperature, torch.arange(len(components), device=prediction.device)
    )
    regression = torch.stack([row[0] + cosine_weight * (1 - row[1]) for row in components]).mean()
    return regression + contrastive_weight * contrastive


def dense_expression_losses(outputs, batch, grid):
    device = outputs[0].lfc.device
    dense = torch.zeros((), device=device)
    distribution = torch.zeros((), device=device)
    centers = torch.linspace(-grid.max_delta, grid.max_delta, grid.n_bins + 1, device=device)
    centers = (centers[1:] + centers[:-1]) / 2
    for output, group in zip(outputs, batch.groups, strict=True):
        folds = group.fold_changes.to(device)
        valid = torch.isfinite(folds) & (folds > 0)
        weight = len(group.rows) / len(batch.dataset_ids)
        if valid.any():
            labels = torch.log2(folds[valid]).clamp(-4, 4)
            dense = dense + weight * F.huber_loss(
                output.lfc[valid].float(), labels.float(), delta=0.5
            )
        delta = group.delta_p.to(device)
        valid = torch.isfinite(delta)
        if valid.any():
            prediction = (output.dp_logits.float().softmax(-1) * centers).sum(-1)
            distribution = distribution + weight * F.huber_loss(
                prediction[valid], delta[valid].float(), delta=0.25
            )
    return dense, distribution
