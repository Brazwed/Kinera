"""Compatibility re-exports for the reorganized cross-entropy APIs."""

from kinera.nn.losses.cross_entropy import (
    _cross_entropy_gradient_validated,
    _cross_entropy_validated,
    _softmax_validated,
    cross_entropy,
    cross_entropy_gradient,
    softmax,
)

__all__ = ["cross_entropy", "cross_entropy_gradient", "softmax"]
