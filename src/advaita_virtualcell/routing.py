"""Control-only background routing for a specialist amplitude readout."""

from dataclasses import dataclass

import numpy as np

from .io import canonical_names


@dataclass
class BackgroundRouter:
    genes: np.ndarray
    names: np.ndarray
    profiles: np.ndarray
    specialist: str
    expression_gate: float = 1e-5

    def __post_init__(self):
        canonical_names(self.genes, "router genes")
        canonical_names(self.names, "reference names")
        self.profiles = np.asarray(self.profiles, dtype=float)
        if self.profiles.shape != (len(self.names), len(self.genes)):
            raise ValueError("Reference profiles do not match the declared axes")
        if not np.isfinite(self.profiles).all() or np.any(self.profiles < 0):
            raise ValueError("Reference profiles must be finite and nonnegative")
        if not np.allclose(self.profiles.sum(1), 1.0):
            raise ValueError("Each reference profile must sum to one")
        if self.specialist not in self.names:
            raise ValueError("The specialist is absent from the reference backgrounds")
        self.gate = self.profiles.max(0) >= self.expression_gate
        if self.gate.sum() < 2:
            raise ValueError("Too few genes pass the reference expression gate")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["genes"],
                data["names"],
                data["profiles"],
                str(data["specialist"].item()),
                float(data["expression_gate"]) if "expression_gate" in data else 1e-5,
            )

    def save(self, path):
        np.savez_compressed(
            path,
            genes=self.genes,
            names=self.names,
            profiles=self.profiles,
            specialist=self.specialist,
            expression_gate=self.expression_gate,
        )

    def route(self, pooled_control, genes):
        if not np.array_equal(canonical_names(genes), canonical_names(self.genes)):
            raise ValueError("Router and query gene order do not match")
        query = np.asarray(pooled_control, dtype=float)
        if query.shape != (len(self.genes),) or not np.isfinite(query).all() or np.any(query < 0):
            raise ValueError("Invalid query control profile")
        if not np.isclose(query.sum(), 1.0):
            raise ValueError("Query control profile must sum to one")
        reference = np.log1p(5e4 * self.profiles[:, self.gate])
        reference -= reference.mean(1, keepdims=True)
        query = np.log1p(5e4 * query[self.gate])
        query -= query.mean()
        norms = np.linalg.norm(reference, axis=1)
        if np.any(norms <= 0) or np.linalg.norm(query) <= 0:
            raise ValueError("Pearson routing is undefined for a constant profile")
        scores = (reference / norms[:, None]) @ (query / np.linalg.norm(query))
        order = np.argsort(-scores, kind="stable")
        nearest = str(self.names[order[0]])
        return {
            "nearest": nearest,
            "use_specialist": nearest == self.specialist,
            "scores": {str(self.names[i]): float(scores[i]) for i in order},
        }
