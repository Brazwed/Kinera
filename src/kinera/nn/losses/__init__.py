"""Validated loss APIs and backend-independent multitask loss composition."""

from .binary_cross_entropy import (
    binary_cross_entropy_with_logits,
    binary_cross_entropy_with_logits_gradient,
)
from .cross_entropy import cross_entropy, cross_entropy_gradient, softmax
from .multitask import MultiTaskLoss

__all__ = [
    "MultiTaskLoss",
    "binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits_gradient",
    "cross_entropy",
    "cross_entropy_gradient",
    "softmax",
]
