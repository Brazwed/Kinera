"""Compatibility imports for the original training.loop module."""

from kinera.training.common import Sample
from kinera.training.config import TrainingConfig, TrainingHistory, TrainingResult
from kinera.training.evaluator import evaluate
from kinera.training.trainer import fit, train_epoch

__all__ = ["Sample", "TrainingConfig", "TrainingHistory", "TrainingResult", "evaluate", "fit", "train_epoch"]
