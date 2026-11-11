"""Full-coverage foundation training followed by laboratory-balanced updates."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..backend import autocast, configure_assets, select_device, subset_sample
from ..evidence import install_evidence_extension
from ..io import file_sha256, read_json, resolve_path, write_json
from .losses import dense_expression_losses


def laboratory(dataset, context):
    if dataset == "replogle" or context in {
        "k562_gwps",
        "k562_essential",
        "rpe1",
        "nadig_jurkat",
        "nadig_hepg2",
    }:
        return "Weissman"
    if dataset == "orion" or context.startswith("orion_"):
        return "Xaira"
    if dataset == "arc_vcc_25" or context == "h1":
        return "Arc"
    if dataset in {"jiang", "tahoe"}:
        return {"jiang": "Satija_Jiang", "tahoe": "Tahoe"}[dataset]
    if dataset == "local_cd4" or context.startswith("cd4_"):
        return "Marson"
    if context in {"kolf", "dld1"}:
        return {"kolf": "Yeo", "dld1": "DLD1"}[context]
    raise ValueError(
        f"No laboratory mapping for {dataset}/{context}; add it to laboratory_by_context"
    )


class RowCycle:
    def __init__(self, dataset, rng, tahoe_weight, overrides=None):
        self.rng = rng
        self.groups = defaultdict(list)
        self.overrides = overrides or {}
        for index, row in enumerate(dataset.rows):
            name = dataset.dirs[row.dir_index].dataset
            lab = (
                self.overrides[row.context]
                if row.context in self.overrides
                else laboratory(name, row.context)
            )
            self.groups[(lab, row.dir_index, row.context)].append(index)
        self.keys = list(self.groups)
        counts = defaultdict(int)
        for lab, _, _ in self.keys:
            counts[lab] += 1
        others = (1 - tahoe_weight) / max(len(counts) - 1, 1)
        weights = {lab: tahoe_weight if lab == "Tahoe" else others for lab in sorted(counts)}
        self.probabilities = np.array([weights[key[0]] / counts[key[0]] for key in self.keys])
        if not len(self.keys) or self.probabilities.sum() <= 0:
            raise ValueError("Training groups must have positive sampling probability")
        self.probabilities /= self.probabilities.sum()
        self.cursors = dict.fromkeys(self.keys, 0)
        for key in self.keys:
            rng.shuffle(self.groups[key])
        self.seen = set()

    def batch(self, count):
        key = self.keys[int(self.rng.choice(len(self.keys), p=self.probabilities))]
        group = self.groups[key]
        positions = []
        for _ in range(count):
            cursor = self.cursors[key]
            if cursor == len(group):
                self.rng.shuffle(group)
                cursor = 0
            index = group[cursor]
            self.cursors[key] = cursor + 1
            self.seen.add(index)
            positions.append(index)
        return positions


def average_precision(labels, probabilities):
    if not labels.any():
        return None
    order = np.argsort(-probabilities, kind="stable")
    labels, probabilities = labels[order], probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(probabilities)), len(probabilities) - 1]
    true_positives = np.cumsum(labels)[ends]
    return float(
        np.sum(np.diff(np.r_[0.0, true_positives / labels.sum()]) * true_positives / (ends + 1))
    )


def train_foundation(path):
    from pie.config import TrainConfig
    from pie.data.datamodule import DataStats, PieDataModule
    from pie.data.dataset import collate
    from pie.model import PieModel, compute_losses
    from pie.predict import _batch_to

    path = Path(path).resolve()
    recipe = read_json(path)
    output = resolve_path(recipe["output"], path.parent)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = read_json(resolve_path(recipe["pie_config"], path.parent))
    raw["data"].update(configure_assets(resolve_path(recipe["assets"], path.parent)))
    raw.update(run_dir=str(output), experiment_name="Advaita_VirtualCell_Model")
    config = TrainConfig.model_validate(raw)
    install_evidence_extension()
    device = select_device(recipe.get("device", "auto"))
    seed = config.seed
    torch.set_num_threads(int(recipe.get("threads", 4)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    pinned = None
    if recipe.get("data_stats"):
        values = read_json(resolve_path(recipe["data_stats"], path.parent))
        pinned = DataStats.model_validate(values.get("stats", values))
    module = PieDataModule(config.data, output, pinned, seed=seed)
    module.setup_stats(0)
    module.setup("fit")
    options = {
        "complete_pass": True,
        "batch_size": 32,
        "genes_per_row": 768,
        "balanced_steps": 1600,
        "tahoe_weight": 0.20,
        "dense_lfc_weight": 0.25,
        "validate_every": 500,
        "precision": "auto",
        **recipe.get("training", {}),
    }
    coverage = []
    if options["complete_pass"]:
        groups = defaultdict(list)
        for index, row in enumerate(module.train_dataset.rows):
            groups[(row.dir_index, row.context)].append(index)
        random = np.random.default_rng(seed)
        for group in groups.values():
            random.shuffle(group)
            coverage.extend(
                [
                    group[i : i + options["batch_size"]]
                    for i in range(0, len(group), options["batch_size"])
                ]
            )
        random.shuffle(coverage)
    total_steps = len(coverage) + options["balanced_steps"]
    if total_steps < 1:
        raise ValueError("Foundation training needs at least one optimization step")
    config = config.model_copy(
        update={
            "trainer": config.trainer.model_copy(update={"max_steps": total_steps}),
            "data": config.data.model_copy(update={"batch_size": options["batch_size"]}),
        }
    )
    model = PieModel(config.model, module.stats, module.source_dims, module.gene_query_text()).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    )
    rng = np.random.default_rng(seed)
    sampler = RowCycle(
        module.train_dataset, rng, options["tahoe_weight"], recipe.get("laboratory_by_context")
    )
    validation = []
    for index in range(len(module.val_dataset)):
        sample = module.val_dataset[index]
        columns = np.sort(
            np.random.default_rng(seed + index).choice(
                len(sample["gene_ids"]),
                min(options["genes_per_row"], len(sample["gene_ids"])),
                replace=False,
            )
        )
        validation.append(subset_sample(sample, columns, include_context_text=True))

    def validate():
        model.eval()
        by_lab = defaultdict(list)
        with torch.inference_mode(), autocast(device, options["precision"]):
            for sample in validation:
                batch = _batch_to(collate([sample], module.source_dims), device, None)
                prediction = model(batch)[0]
                probabilities = (
                    (prediction.de_logits.float() / config.model.temperature.de)
                    .softmax(-1)[0, :, 1]
                    .cpu()
                    .numpy()
                )
                lfc = prediction.lfc[0].float().cpu().numpy()
                labels = batch.groups[0]
                tested, significant = (
                    labels.tested[0].cpu().numpy(),
                    labels.de_mask[0].cpu().numpy(),
                )
                folds = labels.fold_changes[0].cpu().numpy()
                valid = np.isfinite(folds) & (folds > 0)
                score = (
                    average_precision(significant[tested], probabilities[tested])
                    if tested.any()
                    else None
                )
                mae = (
                    float(np.abs(lfc[valid] - np.clip(np.log2(folds[valid]), -4, 4)).mean())
                    if valid.any()
                    else None
                )
                name = module.dirs[sample["dir_index"]].dataset
                overrides = recipe.get("laboratory_by_context", {})
                lab = (
                    overrides[sample["context"]]
                    if sample["context"] in overrides
                    else laboratory(name, sample["context"])
                )
                by_lab[lab].append(dict(ap=score, lfc_mae=mae))
        if not by_lab:
            return None
        labs = {
            lab: {
                key: float(np.mean([item[key] for item in items if item[key] is not None]))
                for key in ("ap", "lfc_mae")
                if any(item[key] is not None for item in items)
            }
            for lab, items in by_lab.items()
        }
        ap = [item["ap"] for item in labs.values() if "ap" in item]
        mae = [item["lfc_mae"] for item in labs.values() if "lfc_mae" in item]
        if not ap or not mae:
            raise ValueError("Foundation validation requires eligible AP and LFC MAE observations")
        return dict(
            ap=float(np.mean(ap)),
            lfc_mae=float(np.mean(mae)),
            selection_score=float(np.mean(ap) - 0.10 * np.mean(mae)),
            laboratories=labs,
        )

    def save(name, step, metric):
        payload = {
            "state_dict": {
                "model." + key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "pie": {
                "format_version": 1,
                "config": config.model_dump(mode="json"),
                "data_stats": module.stats.model_dump(mode="json"),
            },
            "global_step": step,
            "selection_metrics": metric,
            "optimizer": optimizer.state_dict(),
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "sampler_seen": sorted(sampler.seen),
        }
        temporary = output / (name + ".partial")
        torch.save(payload, temporary)
        temporary.replace(output / name)

    write_json(output / "config.json", config.model_dump(mode="json"))
    write_json(
        output / "recipe.json",
        {**options, "coverage_steps": len(coverage), "total_steps": total_steps},
    )
    best, best_step, history = -math.inf, 0, []
    warmup = max(int(total_steps * 0.08), 1)
    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        positions = (
            coverage[step - 1] if step <= len(coverage) else sampler.batch(options["batch_size"])
        )
        sampler.seen.update(positions)
        raw_samples = [module.train_dataset[index] for index in positions]
        n_genes = len(raw_samples[0]["gene_ids"])
        columns = np.sort(
            rng.choice(n_genes, min(options["genes_per_row"], n_genes), replace=False)
        )
        samples = [
            subset_sample(sample, columns, include_context_text=True) for sample in raw_samples
        ]
        batch = _batch_to(collate(samples, module.source_dims), device, None)
        progress = max(0, step - warmup) / max(total_steps - warmup, 1)
        rate = (
            config.optimizer.lr
            * min(step / warmup, 1)
            * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        with autocast(device, options["precision"]):
            outputs = model(batch)
            original = compute_losses(
                outputs, batch, config.model, module.stats.delta_p, temperatures=False
            )["loss"]
            dense, distribution = dense_expression_losses(outputs, batch, module.stats.delta_p)
            loss = original + options["dense_lfc_weight"] * dense + 0.2 * distribution
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite foundation training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite foundation gradient")
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(
                dict(
                    step=step,
                    loss=float(loss.detach()),
                    learning_rate=rate,
                    unique_train_rows=len(sampler.seen),
                ),
                flush=True,
            )
        if (validation and step % options["validate_every"] == 0) or step in {
            len(coverage),
            total_steps,
        }:
            metric = validate()
            score = metric["selection_score"] if metric else float(step)
            history.append(dict(step=step, **(metric or {})))
            if score > best and step >= len(coverage):
                best, best_step = score, step
                save("best.ckpt", step, metric)
            save("last.ckpt", step, metric)
            write_json(output / "validation.json", history)
    if options["complete_pass"] and len(sampler.seen) != len(module.train_dataset):
        raise ArithmeticError("The required full training pass is incomplete")
    result = dict(
        complete=True,
        best_step=best_step,
        best_score=best,
        complete_pass=options["complete_pass"],
        unique_train_rows=len(sampler.seen),
        total_steps=total_steps,
        checkpoint_sha256=file_sha256(output / "best.ckpt"),
    )
    write_json(output / "complete.json", result)
    return result
