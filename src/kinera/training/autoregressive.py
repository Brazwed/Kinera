"""Compatibility import surface for the pre-modular autoregressive API."""

from kinera.language.datasets.adapter import example_to_inputs
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import (
    evaluate_examples,
    evaluate_multitask_examples,
    generate_text,
)
from kinera.training.metrics import _multitask_targets, _safe_ratio
from kinera.training.trainer import train_examples, train_multitask_examples

__all__ = [
    "AutoregressiveModel",
    "evaluate_examples",
    "evaluate_multitask_examples",
    "example_to_inputs",
    "generate_text",
    "train_examples",
    "train_multitask_examples",
]
