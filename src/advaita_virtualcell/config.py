"""Validated configuration for the released inference recipe."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    cells_per_target: int = 400
    top_de_genes: int = 4000
    amplitude_cap: float = 1.5
    amplitude_floor: float = 0.01
    tail_amplitude_cap: float = 0.25
    dense_tail: float = 0.1
    target_retention: float = 0.3
    rank_floor: float = 0.15
    rank_decay_genes: int = 300
    worker_seed: int = 9000
    repair_passes: int = 3
    repair_max_moves: int = 24
    repair_fraction: float = 0.35

    def __post_init__(self):
        if self.cells_per_target < 2 or self.top_de_genes < 0 or self.rank_decay_genes < 2:
            raise ValueError("Invalid cell count, DE budget, or rank schedule")
        if not 0 <= self.amplitude_floor <= self.amplitude_cap:
            raise ValueError("Amplitude bounds must be ordered and nonnegative")
        if not 0 <= self.tail_amplitude_cap <= self.amplitude_cap:
            raise ValueError("Tail amplitude must not exceed the amplitude cap")
        for name in ("dense_tail", "target_retention", "rank_floor", "repair_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.repair_passes, self.repair_max_moves, self.worker_seed) < 0:
            raise ValueError("Repair counts and seed must be nonnegative")
