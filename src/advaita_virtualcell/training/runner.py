"""Source amplitude, contrastive bulk and specialist adaptation recipes."""

import copy
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..backend import (
    autocast,
    configure_assets,
    construct_model,
    load_model_checkpoint,
    select_device,
)
from ..evidence import install_evidence_extension, lineage
from ..io import file_sha256, read_json, resolve_path, write_json
from .data import SourceData
from .losses import (
    bulk_components,
    contrastive_bulk_loss,
    dense_expression_losses,
    deployed_amplitude,
    significant_nmae,
    specialist_loss,
)
from .selection import bulk_candidate, specialist_candidate

READOUT_PREFIXES = ("lfc_head.", "evidence_output_adapters.lfc.")
HEAD_PREFIXES = ("output_head.", "lfc_head.", "delta_p_head.", "evidence_output_adapters.")
DEFAULTS = {
    "amplitude": dict(
        steps=1200,
        seed=20261029,
        batch_size=4,
        genes_per_row=1024,
        learning_rate=1e-5,
        backbone_learning_rate=2e-6,
        validate_every=300,
    ),
    "bulk": dict(
        steps=1200,
        seed=20261029,
        batch_size=8,
        genes_per_row=1024,
        learning_rate=5e-5,
        validate_every=300,
    ),
    "specialist": dict(
        steps=600,
        seed=20261101,
        batch_size=4,
        genes_per_row=2048,
        learning_rate=2e-5,
        validate_every=100,
        near_batch_size=2,
    ),
}


def set_trainable(model, stage):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage == "amplitude" or name.startswith(READOUT_PREFIXES))
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _labels(group, batch, samples, index, device):
    row = int(group.rows[index])
    sample = samples[row]
    truth = (
        sample["lfc_label"].to(device)
        if "lfc_label" in sample
        else torch.log2(group.fold_changes[index].float().clamp_min(1e-30))
    )
    valid = group.tested[index].clone() & torch.isfinite(truth)
    own = int(batch.target_gene[row])
    if own >= 0:
        valid[own] = False
    precision = sample["precision"].to(device) if "precision" in sample else torch.ones_like(truth)
    return truth, valid, group.de_mask[index] & valid, precision


def _bulk_rows(outputs, batch, samples, device):
    rows = []
    for output, group in zip(outputs, batch.groups, strict=True):
        for index, row in enumerate(group.rows.tolist()):
            sample = samples[row]
            values = bulk_components(
                output.lfc[index],
                sample["bulk_control"].to(device),
                sample["bulk_truth"].to(device),
                sample["bulk_available"].to(device),
                int(batch.target_gene[row]),
            )
            rows.append((sample["context"], values))
    return rows


def _bulk_metrics(rows):
    values, vectors = defaultdict(list), defaultdict(list)
    for context, (mse, cosine, prediction, truth, mask) in rows:
        values[lineage(context)].append([float(mse), float(cosine)])
        vectors[context].append(
            (prediction.float().cpu().numpy(), truth.float().cpu().numpy(), mask.cpu().numpy())
        )
    parts = {key: np.mean(value, axis=0).tolist() for key, value in values.items()}
    average = np.mean(list(parts.values()), axis=0)
    retrieval = []
    for items in vectors.values():
        if len(items) < 2:
            continue
        common = np.stack([item[2] for item in items]).all(0)
        prediction = np.stack([item[0] for item in items])[:, common]
        truth = np.stack([item[1] for item in items])[:, common]
        similarity = (
            prediction
            @ truth.T
            / (
                np.linalg.norm(prediction, axis=1)[:, None] * np.linalg.norm(truth, axis=1)[None]
                + 1e-12
            )
        )
        context_scores = []
        for index in range(len(items)):
            others = np.delete(similarity[index], index)
            context_scores.append(
                float(
                    (
                        (others < similarity[index, index]).sum()
                        + 0.5 * (others == similarity[index, index]).sum()
                    )
                    / len(others)
                )
            )
        retrieval.append(float(np.mean(context_scores)))
    if not retrieval:
        raise ValueError("Bulk validation needs at least two targets in one context")
    return dict(
        mse=float(average[0]),
        cosine=float(average[1]),
        retrieval=float(np.mean(retrieval)),
        by_lineage=parts,
    )


class Trainer:
    def __init__(self, recipe, root):
        from pie.data.dataset import collate
        from pie.predict import _batch_to

        self.stage = recipe["stage"]
        if self.stage not in DEFAULTS:
            raise ValueError(f"Unknown training stage: {self.stage}")
        self.options = {**DEFAULTS[self.stage], **recipe.get("training", {})}
        self.options.setdefault("weight_decay", 0.01)
        self.options.setdefault("precision", "auto")
        self.device = select_device(recipe.get("device", "auto"))
        torch.set_num_threads(int(recipe.get("threads", 2)))
        # Source splits are fixed independently of the specialist optimization seed.
        torch.manual_seed(20261029)
        overrides = configure_assets(resolve_path(recipe["assets"], root))
        install_evidence_extension()
        self.loaded = load_model_checkpoint(resolve_path(recipe["checkpoint"], root), overrides)
        self.near = None
        excluded = []
        if self.stage == "specialist":
            with np.load(resolve_path(recipe["near_labels"], root), allow_pickle=False) as archive:
                self.near = {key: archive[key] for key in archive.files}
            excluded = self.near["targets"][self.near["is_validation"]]
        bulk_dir = resolve_path(recipe["source_bulk"], root) if self.stage != "amplitude" else None
        data_checkpoint = self.loaded
        self.direction_loaded = None
        if self.stage == "specialist":
            self.direction_loaded = load_model_checkpoint(
                resolve_path(recipe["direction_checkpoint"], root), overrides
            )
            data_checkpoint = self.direction_loaded
        self.data = SourceData(
            data_checkpoint,
            bulk_dir=bulk_dir,
            exclude_targets=excluded,
            source_datasets=recipe.get("source_datasets", ["local_wilcoxon", "local_cd4"]),
        )
        self.model = construct_model(self.loaded, self.data.module, self.device)
        self.direction = self.teacher = None
        if self.stage == "specialist":
            self.direction = construct_model(self.direction_loaded, self.data.module, self.device)
            self.teacher = copy.deepcopy(self.model).eval()
            for fixed in (self.direction, self.teacher):
                for parameter in fixed.parameters():
                    parameter.requires_grad_(False)
            self.near_samples = self.data.near_samples(self.near)
            self.near_train = np.flatnonzero(~self.near["is_validation"])
            self.near_validation = np.flatnonzero(self.near["is_validation"])
            if len(self.near_train) < self.options["near_batch_size"] or not len(
                self.near_validation
            ):
                raise ValueError(
                    "Specialist adaptation requires enough training and held-out targets"
                )
        self.parameters = set_trainable(self.model, self.stage)
        rate = self.options["learning_rate"]
        if self.stage == "amplitude":
            groups = [
                dict(
                    params=[
                        p for n, p in self.model.named_parameters() if n.startswith(HEAD_PREFIXES)
                    ],
                    lr=rate,
                ),
                dict(
                    params=[
                        p
                        for n, p in self.model.named_parameters()
                        if not n.startswith(HEAD_PREFIXES)
                    ],
                    lr=self.options["backbone_learning_rate"],
                ),
            ]
        else:
            groups = self.parameters
        self.optimizer = torch.optim.AdamW(
            groups, lr=rate, weight_decay=self.options["weight_decay"]
        )
        seed = self.options["seed"]
        if self.stage == "specialist":
            torch.manual_seed(seed)
        self.data.rng = np.random.default_rng(seed)
        validation_seed = {"amplitude": 20261029, "bulk": 20261030, "specialist": 20261101}[
            self.stage
        ]
        self.validation = self.data.validation(
            self.options["genes_per_row"], seed=validation_seed, shared=self.stage != "amplitude"
        )
        self.batch = lambda samples: _batch_to(
            collate(samples, self.data.module.source_dims), self.device, None
        )

    def evaluate_samples(self, samples):
        from pie.model import readout

        self.model.eval()
        values, bulk_rows = defaultdict(list), []
        with torch.no_grad(), autocast(self.device, self.options["precision"]):
            for sample in samples:
                batch = self.batch([sample])
                outputs = self.model(batch)
                if self.stage == "bulk":
                    bulk_rows.extend(_bulk_rows(outputs, batch, [sample], self.device))
                    continue
                direction = self.direction(batch) if self.direction is not None else None
                for group_index, (output, group) in enumerate(
                    zip(outputs, batch.groups, strict=True)
                ):
                    prediction = (
                        deployed_amplitude(output.lfc, direction[group_index].lfc)
                        if direction is not None
                        else readout(
                            output, self.loaded.config.model, self.loaded.stats.delta_p
                        ).lfc
                    )
                    for index in range(len(group.rows)):
                        truth, _, significant, _ = _labels(
                            group, batch, [sample], index, self.device
                        )
                        score = significant_nmae(prediction[index], truth, significant)
                        if score is not None:
                            values[lineage(sample["context"])].append(float(score))
        if self.stage == "bulk":
            return _bulk_metrics(bulk_rows)
        if not values:
            raise ValueError("Validation has no targets with at least ten significant genes")
        parts = {key: float(np.mean(value)) for key, value in values.items()}
        return dict(
            nmae=float(np.mean(list(parts.values()))),
            by_lineage=parts,
            eligible_rows={key: len(value) for key, value in values.items()},
        )

    def evaluate(self, step):
        if self.stage == "specialist":
            return dict(
                step=step,
                source=self.evaluate_samples(self.validation),
                near=self.evaluate_samples([self.near_samples[i] for i in self.near_validation]),
            )
        return dict(step=step, **self.evaluate_samples(self.validation))

    def train_step(self, step):
        from pie.model import compute_losses

        if self.stage == "specialist" and step % 4 == 1:
            chosen = self.data.rng.choice(
                self.near_train, self.options["near_batch_size"], replace=False
            )
            samples = [self.near_samples[int(index)] for index in chosen]
        else:
            samples = self.data.batch(self.options["batch_size"], self.options["genes_per_row"])
        batch = self.batch(samples)
        self.model.train(self.stage == "amplitude")
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, self.options["precision"]):
            outputs = self.model(batch)
            if self.stage == "bulk":
                loss = contrastive_bulk_loss(
                    [row[1] for row in _bulk_rows(outputs, batch, samples, self.device)]
                )
            elif self.stage == "specialist":
                with torch.no_grad():
                    direction, teacher = self.direction(batch), self.teacher(batch)
                terms = []
                for output, base, old, group in zip(
                    outputs, direction, teacher, batch.groups, strict=True
                ):
                    for index in range(len(group.rows)):
                        labels = _labels(group, batch, samples, index, self.device)
                        terms.append(
                            specialist_loss(
                                output.lfc[index], base.lfc[index], old.lfc[index], *labels
                            )
                        )
                loss = torch.stack(terms).mean()
            else:
                terms = []
                for output, group in zip(outputs, batch.groups, strict=True):
                    for index in range(len(group.rows)):
                        truth, _, significant, _ = _labels(
                            group, batch, samples, index, self.device
                        )
                        term = significant_nmae(
                            output.lfc[index], truth, significant, training=True
                        )
                        if term is not None:
                            terms.append(term)
                nmae = torch.stack(terms).mean() if terms else outputs[0].lfc.sum() * 0
                dense, distribution = dense_expression_losses(
                    outputs, batch, self.loaded.stats.delta_p
                )
                original = compute_losses(
                    outputs,
                    batch,
                    self.loaded.config.model,
                    self.loaded.stats.delta_p,
                    temperatures=False,
                )["loss"]
                loss = original + 0.5 * nmae + 0.25 * dense + 0.2 * distribution
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        return float(loss.detach())

    def save(self, destination, step, metrics):
        payload = {
            "state_dict": {
                "model." + key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "pie": {
                "format_version": 1,
                "config": self.loaded.config.model_dump(mode="json"),
                "data_stats": self.loaded.stats.model_dump(mode="json"),
            },
            "global_step": step,
            "adaptation_stage": self.stage,
            "validation": metrics,
        }
        temporary = destination.with_suffix(".partial.ckpt")
        torch.save(payload, temporary)
        temporary.replace(destination)


def train_from_config(path):
    path = Path(path).resolve()
    recipe = read_json(path)
    output = resolve_path(recipe["output"], path.parent)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(recipe, path.parent)
    protocol = {
        "stage": trainer.stage,
        "training": trainer.options,
        "checkpoint_sha256": file_sha256(resolve_path(recipe["checkpoint"], path.parent)),
        "trainable_parameters": sum(p.numel() for p in trainer.parameters),
        "same_lineage_evidence_excluded": True,
        "context_text": False,
        "context_response": False,
    }
    write_json(output / "protocol.json", protocol)
    baseline = trainer.evaluate(0)
    history = [baseline]
    trainer.save(output / "best.ckpt", 0, baseline)
    selected, best = (
        0,
        1.0
        if trainer.stage == "specialist"
        else baseline["mse" if trainer.stage == "bulk" else "nmae"],
    )
    started = time.monotonic()
    print({"baseline": baseline}, flush=True)
    for step in range(1, int(trainer.options["steps"]) + 1):
        loss = trainer.train_step(step)
        if step % 20 == 0:
            print(
                dict(step=step, loss=loss, seconds=round(time.monotonic() - started, 1)), flush=True
            )
        if step % trainer.options["validate_every"] and step != trainer.options["steps"]:
            continue
        metrics = trainer.evaluate(step)
        if trainer.stage == "specialist":
            selection = specialist_candidate(
                metrics["source"]["nmae"],
                metrics["near"]["nmae"],
                baseline["source"]["nmae"],
                baseline["near"]["nmae"],
            )
            eligible, score = selection["passed"], selection["score"]
            metrics.update(selection)
        elif trainer.stage == "bulk":
            eligible = bulk_candidate(
                metrics["mse"],
                metrics["cosine"],
                metrics["retrieval"],
                baseline["cosine"],
                baseline["retrieval"],
            )
            score = metrics["mse"]
        else:
            eligible, score = True, metrics["nmae"]
        metrics.update(eligible=eligible, selection_score=score)
        history.append(metrics)
        if eligible and score < best:
            selected, best = step, score
            trainer.save(output / "best.ckpt", step, metrics)
        write_json(output / "history.json", history)
        print(metrics, flush=True)
    result = dict(
        complete=True,
        selected_step=selected,
        advanced=selected > 0,
        selection_score=best,
        checkpoint_sha256=file_sha256(output / "best.ckpt"),
    )
    write_json(output / "complete.json", result)
    return result
