"""Teste de cinco minutos para uma expressão matemática fixa."""

from __future__ import annotations

import math
import time
from pathlib import Path

from kinera.nn.layers.neuron import Neuron

DURATION_SECONDS = 300.0
LEARNING_RATE = 1e-7
REPORT_INTERVAL_SECONDS = 30.0
LOG_PATH = Path(__file__).resolve().with_name("expression_5min.log")

BASE_INPUTS = [37.5, 18.2, 7.3, 5.7, 2.9, 987.65, 3.14, 2.7, 1234.5]


def expression(values: list[float]) -> float:
    first, second, third, fourth, fifth, sixth, seventh, eighth, ninth = values
    return (
        ((first**2 - second * third) / (fourth + fifth))
        + math.sqrt(sixth)
        - (seventh**3 / eighth)
        + math.log(ninth)
    )


def main() -> None:
    target = expression(BASE_INPUTS)
    neuron = Neuron(input_size=len(BASE_INPUTS), seed=7, activation="linear")

    started_at = time.monotonic()
    deadline = started_at + DURATION_SECONDS
    next_report = started_at + REPORT_INTERVAL_SECONDS
    steps = 0

    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write("Kinera — teste matemático de 5 minutos\n")
        log.write(f"target={target:.15f}\n")
        log.write(f"learning_rate={LEARNING_RATE}\n")
        log.flush()

        while time.monotonic() < deadline:
            prediction = neuron.forward(BASE_INPUTS)
            error = prediction - target
            neuron.backward(error)
            neuron.step(LEARNING_RATE)
            steps += 1

            now = time.monotonic()
            if now >= next_report:
                elapsed = now - started_at
                loss = 0.5 * error**2
                message = (
                    f"elapsed={elapsed:.1f}s steps={steps} "
                    f"prediction={prediction:.15f} error={error:.15f} "
                    f"loss={loss:.15f}"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                next_report += REPORT_INTERVAL_SECONDS

        final_prediction = neuron.forward(BASE_INPUTS)
        final_error = final_prediction - target
        summary = (
            f"finished elapsed={time.monotonic() - started_at:.3f}s "
            f"steps={steps} target={target:.15f} "
            f"prediction={final_prediction:.15f} error={final_error:.15f}"
        )
        print(summary, flush=True)
        log.write(summary + "\n")

        validation_cases = [
            ("original", BASE_INPUTS),
            ("first_constant_changed", [38.5, *BASE_INPUTS[1:]]),
            ("sqrt_constant_changed", [*BASE_INPUTS[:5], 1000.0, *BASE_INPUTS[6:]]),
            ("cube_constant_changed", [*BASE_INPUTS[:6], 3.0, *BASE_INPUTS[7:]]),
        ]
        for name, inputs in validation_cases:
            expected = expression(inputs)
            prediction = neuron.forward(inputs)
            line = (
                f"validation={name} expected={expected:.15f} "
                f"prediction={prediction:.15f} "
                f"error={prediction - expected:.15f}"
            )
            print(line, flush=True)
            log.write(line + "\n")


if __name__ == "__main__":
    main()
