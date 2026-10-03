"""Semantically named syllable-boundary prediction head."""

from kinera.backends import Backend
from kinera.nn.layers.linear import Linear


class BoundaryHead(Linear):
    """Single-logit linear head for boundary prediction."""

    def __init__(self, input_dim: int, *, seed: int | None = None, backend: Backend | None = None) -> None:
        super().__init__(input_dim, 1, seed=seed, backend=backend)
