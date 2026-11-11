"""Source-target splits and lineage-masked model samples."""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..backend import subset_sample
from ..evidence import LineageEvidence, lineage


def held_out_target(target):
    digest = hashlib.sha256(("source-generalization-" + str(target)).encode()).hexdigest()
    return int(digest[:8], 16) % 5 == 0


class SourceData:
    def __init__(
        self,
        checkpoint,
        *,
        bulk_dir=None,
        seed=20261029,
        source_datasets=("local_wilcoxon", "local_cd4"),
        exclude_targets=(),
    ):
        from pie.data.datamodule import PieDataModule

        self.rng = np.random.default_rng(seed)
        self.module = PieDataModule(checkpoint.config.data, None, checkpoint.stats, seed=seed)
        self.module.setup_stats(0)
        self.module.setup("fit")
        self.dataset = self.module.train_dataset
        self.evidence = LineageEvidence(self.module._evidence_index)
        self.bulk = {}
        self.bulk_rows = {}
        self.bulk_columns = {}
        if bulk_dir is not None:
            for path in sorted(Path(bulk_dir).glob("*.npz")):
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files}
                self.bulk[path.stem] = data
                self.bulk_rows[path.stem] = {str(t): i for i, t in enumerate(data["targets"])}
                genes = {str(g).upper(): i for i, g in enumerate(data["genes"])}
                for index, directory in enumerate(self.module.dirs):
                    if path.stem in directory.meta.context_to_id:
                        self.bulk_columns[(path.stem, index)] = np.array(
                            [genes.get(g.upper(), -1) for g in directory.genes]
                        )
            if not self.bulk:
                raise ValueError("No source bulk archives found")
        self.groups = defaultdict(list)
        validation = defaultdict(list)
        excluded = {str(target).upper() for target in exclude_targets}
        for index, row in enumerate(self.dataset.rows):
            directory = self.module.dirs[row.dir_index]
            if directory.dataset not in source_datasets:
                continue
            if bulk_dir is not None and (
                row.context not in self.bulk_rows
                or row.perturbation not in self.bulk_rows[row.context]
            ):
                continue
            group = lineage(row.context)
            if held_out_target(row.perturbation):
                validation[group].append(index)
            elif row.perturbation.upper() not in excluded:
                self.groups[(group, row.dir_index, row.context)].append(index)
        self.keys = sorted(self.groups)
        self.validation_rows = [
            int(index)
            for group in sorted(validation)
            for index in np.random.default_rng(seed).choice(
                validation[group], min(16, len(validation[group])), replace=False
            )
        ]
        if not self.keys or not self.validation_rows:
            raise ValueError("Source training and validation both require eligible rows")

    def sample(self, index, *, columns=None, seed=None, n_genes=1024):
        sample = self.dataset[index]
        rng = self.rng if seed is None else np.random.default_rng(seed)
        if columns is None:
            columns = np.sort(
                rng.choice(
                    len(sample["gene_ids"]), min(n_genes, len(sample["gene_ids"])), replace=False
                )
            )
        sample = subset_sample(sample, columns)
        sample["evidence"] = {
            key: torch.from_numpy(value)
            for key, value in self.evidence.blocks(
                lineage(sample["context"]), sample["perturbation"], sample["gene_ids"].numpy()
            ).items()
        }
        if self.bulk:
            row = self.dataset.rows[index]
            data = self.bulk[row.context]
            target = self.bulk_rows[row.context][row.perturbation]
            selected = self.bulk_columns[(row.context, row.dir_index)][columns]
            available = selected >= 0
            for key, field in (
                ("bulk_control", "control"),
                ("bulk_truth", "pool"),
                ("ctrl_mean", "features"),
            ):
                values = np.zeros(len(selected), np.float32)
                values[available] = data[field][target, selected[available]]
                sample[key] = torch.from_numpy(values)
            sample["bulk_available"] = torch.from_numpy(available)
        return sample

    def batch(self, count, n_genes):
        key = self.keys[int(self.rng.integers(len(self.keys)))]
        chosen = self.rng.choice(
            self.groups[key], size=count, replace=len(self.groups[key]) < count
        )
        size = len(self.module.dirs[key[1]].genes)
        columns = np.sort(self.rng.choice(size, min(n_genes, size), replace=False))
        return [self.sample(int(index), columns=columns) for index in chosen]

    def validation(self, n_genes=1024, seed=20261030, shared=True):
        if not shared:
            return [
                self.sample(index, seed=seed + position, n_genes=n_genes)
                for position, index in enumerate(self.validation_rows)
            ]
        columns = {
            i: np.sort(
                np.random.default_rng(seed + i).choice(
                    len(directory.genes), min(n_genes, len(directory.genes)), replace=False
                )
            )
            for i, directory in enumerate(self.module.dirs)
        }
        return [
            self.sample(index, columns=columns[self.dataset.rows[index].dir_index])
            for index in self.validation_rows
        ]

    def near_samples(self, archive):
        lookup = {gene.upper(): index for index, gene in enumerate(self.module.gene_axis)}
        canonical = {gene.upper(): gene for gene in self.module.gene_axis}
        columns = np.array(
            [i for i, gene in enumerate(archive["genes"]) if str(gene).upper() in lookup]
        )
        gene_ids = np.array([lookup[str(archive["genes"][i]).upper()] for i in columns])
        genes = archive["genes"][columns].astype(str)
        samples = []
        for index, target in enumerate(archive["targets"].astype(str)):
            tokens = {}
            for name in self.dataset.sources.dims:
                if name == "context_text":
                    continue
                value = self.dataset.sources.tokens(
                    name,
                    context="anon_molm13_amplitude",
                    perturbation=canonical.get(target.upper(), target),
                )
                if value is not None:
                    tokens[name] = torch.from_numpy(value)
            own = np.flatnonzero(np.char.upper(genes) == target.upper())
            truth = archive["lfc"][index, columns].astype(np.float32)
            samples.append(
                {
                    "dir_index": 0,
                    "row": -1,
                    "context": "molm13_amplitude",
                    "perturbation": target,
                    "source_tokens": tokens,
                    "gene_ids": torch.from_numpy(gene_ids),
                    "ctrl_mean": torch.from_numpy(archive["features"][columns].astype(np.float32)),
                    "evidence": {
                        key: torch.from_numpy(value)
                        for key, value in self.evidence.blocks("molm13", target, gene_ids).items()
                    },
                    "target_gene": int(own[0]) if len(own) else -1,
                    "target_gene_idx": int(own[0]) if len(own) else -1,
                    "fold_changes": torch.from_numpy(np.exp2(np.clip(truth, -30, 30))),
                    "de_mask": torch.from_numpy(archive["significant"][index, columns]),
                    "tested": torch.from_numpy(archive["tested"][index, columns]),
                    "delta_p": torch.zeros(len(columns), dtype=torch.float32),
                    "precision": torch.from_numpy(archive["precision"][index, columns]),
                    "lfc_label": torch.from_numpy(truth),
                }
            )
        return samples
