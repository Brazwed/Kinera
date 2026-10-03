"""Adaptador de treino contínuo que reutiliza o ciclo da Parte 1."""

from __future__ import annotations

import random
import time
from pathlib import Path

from kinera.nn.layers.neuron import Neuron
from kinera.training.evaluator import evaluate as evaluate_model
from kinera.training.trainer import train_epoch

DURATION_SECONDS = 300.0
LEARNING_RATE = 0.002
REPORT_INTERVAL_SECONDS = 30.0
BATCH_SIZE = 1_024
LOG_PATH = Path(__file__).resolve().with_name("training_5min.log")
VALIDATION_SAMPLES = [
    ([0.5, 0.5], 3.5),
    ([1.0, 2.0], 9.0),
    ([2.0, 1.0], 8.0),
    ([3.0, 4.0], 19.0),
]


def target_function(x1: float, x2: float) -> float:
    return 2.0 * x1 + 3.0 * x2 + 1.0


def build_samples(
    generator: random.Random,
    count: int,
) -> list[tuple[list[float], float]]:
    """Gera um lote determinístico para a função-alvo."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count deve ser um inteiro")
    if count <= 0:
        raise ValueError("count deve ser positivo")

    samples = []
    for _ in range(count):
        x1 = generator.uniform(0.25, 4.0)
        x2 = generator.uniform(0.25, 4.0)
        samples.append(([x1, x2], target_function(x1, x2)))
    return samples


def validation_report(
    neuron: Neuron,
) -> tuple[float, list[tuple[list[float], float, float]]]:
    """Retorna loss e previsões do conjunto fixo de validação."""
    validation_loss = evaluate_model(neuron, VALIDATION_SAMPLES)
    results = []
    for inputs, expected in VALIDATION_SAMPLES:
        prediction = neuron.forward(inputs)
        results.append((inputs, expected, prediction))
    return validation_loss, results


def main() -> None:
    neuron = Neuron(input_size=2, seed=7)
    neuron.weights = [0.5, 0.5]
    neuron.bias = 0.5
    generator = random.Random(20261001)

    started_at = time.monotonic()
    deadline = started_at + DURATION_SECONDS
    next_report = started_at + REPORT_INTERVAL_SECONDS
    epoch = 0
    steps = 0
    weighted_train_loss = 0.0
    best_validation_loss = float("inf")

    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write("Kinera — treino contínuo de 5 minutos\n")
        log.write(f"learning_rate={LEARNING_RATE}\n")
        log.write(f"batch_size={BATCH_SIZE}\n")
        log.flush()

        while time.monotonic() < deadline:
            samples = build_samples(generator, BATCH_SIZE)
            train_loss = train_epoch(
                neuron,
                samples,
                LEARNING_RATE,
                shuffle=True,
                rng=generator,
            )
            epoch += 1
            steps += len(samples)
            weighted_train_loss += train_loss * len(samples)

            now = time.monotonic()
            if now >= next_report:
                elapsed = now - started_at
                validation_loss, _ = validation_report(neuron)
                best_validation_loss = min(best_validation_loss, validation_loss)
                message = (
                    f"elapsed={elapsed:.1f}s epochs={epoch} steps={steps} "
                    f"mean_train_loss={weighted_train_loss / steps:.12f} "
                    f"validation_loss={validation_loss:.12f} "
                    f"weights={[round(value, 8) for value in neuron.weights]} "
                    f"bias={neuron.bias:.8f}"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                next_report += REPORT_INTERVAL_SECONDS

        elapsed = time.monotonic() - started_at
        validation_loss, results = validation_report(neuron)
        best_validation_loss = min(best_validation_loss, validation_loss)
        summary = (
            f"finished elapsed={elapsed:.3f}s epochs={epoch} steps={steps} "
            f"mean_train_loss={weighted_train_loss / steps:.12f} "
            f"best_validation_loss={best_validation_loss:.12f} "
            f"final_validation_loss={validation_loss:.12f} "
            f"weights={[round(value, 12) for value in neuron.weights]} "
            f"bias={neuron.bias:.12f}"
        )
        print(summary, flush=True)
        log.write(summary + "\n")
        for inputs, expected, prediction in results:
            line = (
                f"validation inputs={inputs} expected={expected:.12f} "
                f"prediction={prediction:.12f} "
                f"error={prediction - expected:.12f}"
            )
            print(line, flush=True)
            log.write(line + "\n")


if __name__ == "__main__":
    main()
