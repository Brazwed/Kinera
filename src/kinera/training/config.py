"""Configuration and result records for single-neuron training."""

from __future__ import annotations

from dataclasses import dataclass

from kinera.training.common import Sample, _validate_number


@dataclass(frozen=True)
class TrainingConfig:
    """Parâmetros explícitos de uma execução de treinamento."""

    learning_rate: float
    max_epochs: int
    patience: int
    min_delta: float = 0.0
    shuffle: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_number(self.learning_rate, "learning_rate")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate deve ser positivo")
        if isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int):
            raise TypeError("max_epochs deve ser um inteiro")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs deve ser positivo")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise TypeError("patience deve ser um inteiro")
        if self.patience < 0:
            raise ValueError("patience não pode ser negativo")
        _validate_number(self.min_delta, "min_delta")
        if self.min_delta < 0.0:
            raise ValueError("min_delta não pode ser negativo")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle deve ser booleano")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed deve ser um inteiro ou None")


@dataclass(frozen=True)
class TrainingHistory:
    """Métricas e motivo de parada de uma execução."""

    train_losses: list[float]
    validation_losses: list[float]
    steps: int
    best_epoch: int
    best_validation_loss: float
    stopped_reason: str

    @property
    def epochs_completed(self) -> int:
        return len(self.train_losses)


@dataclass(frozen=True)
class TrainingResult:
    """Resultado completo, incluindo os parâmetros restaurados do melhor ponto."""

    history: TrainingHistory
    final_weights: list[float]
    final_bias: float

__all__ = ["Sample", "TrainingConfig", "TrainingHistory", "TrainingResult"]
