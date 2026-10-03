"""Executor do experimento de treinamento com múltiplas seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from kinera.training.checkpoint import load_neuron, save_neuron
from kinera.experiments.synthetic_data import generate_samples, split_samples
from kinera.experiments.runner import (
    MultiSeedSummary,
    format_multi_seed_report,
    run_single_seed,
    run_multi_seed,
)
from kinera.training.config import TrainingConfig

DEFAULT_SEEDS = (1, 7, 42, 123, 2026)
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_DATA_SEED = 7
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_ACTIVATION = "leaky_relu"
DEFAULT_NEGATIVE_SLOPE = 0.01


def run_experiment(
    *,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    data_seed: int = DEFAULT_DATA_SEED,
    config: TrainingConfig | None = None,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    activation: str = DEFAULT_ACTIVATION,
    negative_slope: float = DEFAULT_NEGATIVE_SLOPE,
    checkpoint_path: str | Path | None = None,
    checkpoint_seed: int | None = None,
) -> MultiSeedSummary:
    """Gera um dataset compartilhado e executa todas as seeds."""
    samples = generate_samples(sample_count, seed=data_seed)
    train_samples, validation_samples = split_samples(
        samples,
        validation_ratio=validation_ratio,
        seed=data_seed,
    )
    effective_config = config or TrainingConfig(
        learning_rate=0.01,
        max_epochs=500,
        patience=20,
        min_delta=1e-10,
        shuffle=True,
        seed=42,
    )
    if checkpoint_path is not None:
        if checkpoint_seed is None:
            if len(seeds) != 1:
                raise ValueError(
                    "checkpoint_seed é obrigatório quando há várias seeds"
                )
            checkpoint_seed = seeds[0]
        if checkpoint_seed not in seeds:
            raise ValueError("checkpoint_seed deve estar em seeds")

    summary = run_multi_seed(
        seeds,
        train_samples,
        validation_samples,
        effective_config,
        convergence_threshold=convergence_threshold,
        activation=activation,
        negative_slope=negative_slope,
    )
    if checkpoint_path is not None:
        checkpoint_result, checkpoint_model = run_single_seed(
            checkpoint_seed,
            train_samples,
            validation_samples,
            effective_config,
            convergence_threshold=convergence_threshold,
            activation=activation,
            negative_slope=negative_slope,
        )
        if checkpoint_model is None or checkpoint_result.status == "error":
            raise RuntimeError(
                f"não foi possível treinar a seed do checkpoint: "
                f"{checkpoint_result.error}"
            )
        saved_path = save_neuron(
            checkpoint_path,
            checkpoint_model,
            training_state={
                "epoch": checkpoint_result.epochs_completed,
                "steps": checkpoint_result.steps,
                "best_epoch": checkpoint_result.best_epoch,
                "best_validation_loss": checkpoint_result.best_validation_loss,
                "stopped_reason": checkpoint_result.stopped_reason,
                "config": {
                    "learning_rate": effective_config.learning_rate,
                    "max_epochs": effective_config.max_epochs,
                    "patience": effective_config.patience,
                    "min_delta": effective_config.min_delta,
                    "shuffle": effective_config.shuffle,
                    "seed": effective_config.seed,
                },
            },
            metadata={
                "seed": checkpoint_seed,
                "activation": activation,
                "negative_slope": negative_slope,
                "data_seed": data_seed,
                "sample_count": sample_count,
                "validation_ratio": validation_ratio,
            },
        )
        reloaded, _ = load_neuron(saved_path)
        if reloaded.weights != checkpoint_model.weights or reloaded.bias != checkpoint_model.bias:
            raise RuntimeError("checkpoint salvo não reproduz os pesos treinados")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o experimento multi-seed da Kinera."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
    )
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-10)
    parser.add_argument(
        "--activation",
        choices=("relu", "leaky_relu", "linear"),
        default=DEFAULT_ACTIVATION,
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=DEFAULT_NEGATIVE_SLOPE,
    )
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        shuffle=args.shuffle,
        seed=42,
    )
    summary = run_experiment(
        seeds=args.seeds,
        sample_count=args.sample_count,
        validation_ratio=args.validation_ratio,
        data_seed=args.data_seed,
        config=config,
        convergence_threshold=args.convergence_threshold,
        activation=args.activation,
        negative_slope=args.negative_slope,
        checkpoint_path=args.checkpoint_path,
        checkpoint_seed=args.checkpoint_seed,
    )
    print(format_multi_seed_report(summary))
    if args.checkpoint_path is not None:
        print(f"checkpoint={args.checkpoint_path}")


if __name__ == "__main__":
    main()
