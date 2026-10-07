"""Execução reprodutível de experimentos com múltiplas inicializações."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kinera.monitor.metrics.core import safe_ratio
from kinera.nn.layers.neuron import Neuron
from kinera.training.config import Sample, TrainingConfig
from kinera.training.trainer import fit


@dataclass(frozen=True)
class SeedExperimentResult:
    """Resultado de uma execução individual."""

    seed: int
    activation: str
    negative_slope: float
    initial_active_count: int | None
    initial_sample_count: int
    initial_active_fraction: float | None
    status: str
    converged: bool
    epochs_completed: int
    steps: int
    best_epoch: int | None
    stopped_reason: str | None
    best_train_loss: float | None
    best_validation_loss: float | None
    final_weights: list[float]
    final_bias: float | None
    validation_predictions: list[float]
    failure_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MultiSeedSummary:
    """Resultados agregados do experimento multi-seed."""

    results: list[SeedExperimentResult]

    @property
    def converged_count(self) -> int:
        return sum(result.status == "converged" for result in self.results)

    @property
    def not_converged_count(self) -> int:
        return sum(result.status == "not_converged" for result in self.results)

    @property
    def error_count(self) -> int:
        return sum(result.status == "error" for result in self.results)

    @property
    def convergence_rate(self) -> float:
        return safe_ratio(self.converged_count, len(self.results))


def run_single_seed(
    seed: int,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    config: TrainingConfig,
    *,
    convergence_threshold: float = 1e-6,
    activation: str = "relu",
    negative_slope: float = 0.01,
) -> tuple[SeedExperimentResult, Neuron | None]:
    """Treina uma seed e retorna o relatório junto do modelo final."""
    input_size = len(train_samples[0][0])
    try:
        neuron = Neuron(
            input_size=input_size,
            seed=seed,
            activation=activation,
            negative_slope=negative_slope,
        )
        initial_active_count, initial_sample_count = _initial_activity(
            neuron, train_samples
        )
        initial_active_fraction = (
            None
            if initial_active_count is None
            else safe_ratio(initial_active_count, initial_sample_count)
        )
        training_result = fit(
            neuron,
            train_samples,
            validation_samples,
            config,
        )
        best_validation_loss = training_result.history.best_validation_loss
        converged = best_validation_loss <= convergence_threshold
        status = "converged" if converged else "not_converged"
        failure_reason = (
            None if converged else "best_validation_loss_above_threshold"
        )
        predictions = [
            neuron.forward(inputs) for inputs, _ in validation_samples
        ]
        result = SeedExperimentResult(
            seed=seed,
            activation=activation,
            negative_slope=neuron.negative_slope,
            initial_active_count=initial_active_count,
            initial_sample_count=initial_sample_count,
            initial_active_fraction=initial_active_fraction,
            status=status,
            converged=converged,
            epochs_completed=training_result.history.epochs_completed,
            steps=training_result.history.steps,
            best_epoch=training_result.history.best_epoch,
            stopped_reason=training_result.history.stopped_reason,
            best_train_loss=min(training_result.history.train_losses),
            best_validation_loss=best_validation_loss,
            final_weights=training_result.final_weights.copy(),
            final_bias=training_result.final_bias,
            validation_predictions=predictions,
            failure_reason=failure_reason,
        )
        return result, neuron
    except Exception as error:  # Cada seed deve gerar um resultado observável.
        result = SeedExperimentResult(
            seed=seed,
            activation=activation,
            negative_slope=_safe_slope(negative_slope),
            initial_active_count=None,
            initial_sample_count=len(train_samples),
            initial_active_fraction=None,
            status="error",
            converged=False,
            epochs_completed=0,
            steps=0,
            best_epoch=None,
            stopped_reason=None,
            best_train_loss=None,
            best_validation_loss=None,
            final_weights=[],
            final_bias=None,
            validation_predictions=[],
            error=f"{type(error).__name__}: {error}",
        )
        return result, None


def run_multi_seed(
    seeds: list[int] | tuple[int, ...],
    train_samples: list[Sample],
    validation_samples: list[Sample],
    config: TrainingConfig,
    *,
    convergence_threshold: float = 1e-6,
    activation: str = "relu",
    negative_slope: float = 0.01,
) -> MultiSeedSummary:
    """Executa o mesmo treinamento para cada seed, sem descartar falhas."""
    _validate_seeds(seeds)
    if not isinstance(config, TrainingConfig):
        raise TypeError("config deve ser uma instância de TrainingConfig")
    _validate_activation_config(activation, negative_slope)
    _validate_number(convergence_threshold, "convergence_threshold")
    if convergence_threshold < 0.0:
        raise ValueError("convergence_threshold não pode ser negativo")
    if not train_samples:
        raise ValueError("train_samples não pode ser vazio")
    input_size = len(train_samples[0][0])
    if input_size <= 0:
        raise ValueError("as entradas devem possuir pelo menos um valor")

    results = [
        run_single_seed(
            seed,
            train_samples,
            validation_samples,
            config,
            convergence_threshold=convergence_threshold,
            activation=activation,
            negative_slope=negative_slope,
        )[0]
        for seed in seeds
    ]
    return MultiSeedSummary(results=results)


def format_multi_seed_report(summary: MultiSeedSummary) -> str:
    """Formata resultados em texto simples para terminal e logs."""
    if not isinstance(summary, MultiSeedSummary):
        raise TypeError("summary deve ser uma instância de MultiSeedSummary")

    lines = []
    for result in summary.results:
        validation_loss = (
            "n/a"
            if result.best_validation_loss is None
            else f"{result.best_validation_loss:.12f}"
        )
        line = (
            f"seed={result.seed} activation={result.activation} "
            f"initial_active={result.initial_active_count}/"
            f"{result.initial_sample_count} status={result.status} "
            f"epochs={result.epochs_completed} steps={result.steps} "
            f"best_validation_loss={validation_loss}"
        )
        if result.failure_reason is not None:
            line += f" failure_reason={result.failure_reason}"
        if result.error is not None:
            line += f" error={result.error}"
        lines.append(line)
    lines.append(
        f"converged={summary.converged_count} "
        f"not_converged={summary.not_converged_count} "
        f"errors={summary.error_count} "
        f"convergence_rate={summary.convergence_rate:.4f}"
    )
    return "\n".join(lines)


def _validate_seeds(seeds: list[int] | tuple[int, ...]) -> None:
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds deve conter pelo menos uma seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds não pode conter duplicatas")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("cada seed deve ser um inteiro")


def _validate_activation_config(activation: str, negative_slope: float) -> None:
    if activation not in {"relu", "leaky_relu", "linear"}:
        raise ValueError(f"activation desconhecida: {activation}")
    _validate_number(negative_slope, "negative_slope")
    if activation == "leaky_relu" and not 0.0 < negative_slope <= 1.0:
        raise ValueError("negative_slope deve estar entre 0 e 1")


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito")


def _initial_activity(
    neuron: Neuron,
    samples: list[Sample],
) -> tuple[int | None, int]:
    if neuron.activation == "linear":
        return None, len(samples)
    active_count = 0
    for inputs, _ in samples:
        z = sum(
            input_value * weight
            for input_value, weight in zip(inputs, neuron.weights)
        )
        z += neuron.bias
        active_count += z > 0.0
    return active_count, len(samples)


def _safe_slope(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return float(value)
    return 0.01
