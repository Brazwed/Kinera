"""Semantically named character prediction head."""

from kinera.backends import Backend
from kinera.nn.layers.linear import Linear


class CharacterHead(Linear):
    """Linear logits head for character targets."""

    def __init__(self, input_dim: int, vocab_size: int, *, seed: int | None = None, backend: Backend | None = None) -> None:
        super().__init__(input_dim, vocab_size, seed=seed, backend=backend)
