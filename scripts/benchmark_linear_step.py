"""Paired microbenchmark for Linear.step() candidates; never edits production."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn.linear import Linear  # noqa: E402


VARIANTS: dict[str, Callable[[Linear, float], None]]


def _validate_step_contract(layer: Linear, learning_rate: float) -> None:
    layer._validate_number(learning_rate, "learning_rate")
    if learning_rate < 0.0:
        raise ValueError("learning_rate não pode ser negativo")
    if not layer._has_backward:
        raise RuntimeError("step exige um backward anterior")


def step_a(layer: Linear, learning_rate: float) -> None:
    """A: current production implementation."""
    layer.step(learning_rate)


def step_b(layer: Linear, learning_rate: float) -> None:
    """B: cache float(learning_rate) once; keep full list reconstruction."""
    _validate_step_contract(layer, learning_rate)
    rate = float(learning_rate)
    layer.weights = [
        [
            weight - rate * gradient
            for weight, gradient in zip(row, gradient_row)
        ]
        for row, gradient_row in zip(layer.weights, layer.grad_weights)
    ]
    layer.bias = [
        bias - rate * gradient
        for bias, gradient in zip(layer.bias, layer.grad_bias)
    ]
    layer._has_backward = False


def step_c(layer: Linear, learning_rate: float) -> None:
    """C: cache float(learning_rate) once and update weights/biases in place."""
    _validate_step_contract(layer, learning_rate)
    rate = float(learning_rate)
    for row, gradient_row in zip(layer.weights, layer.grad_weights):
        for index, (weight, gradient) in enumerate(zip(row, gradient_row)):
            row[index] = weight - rate * gradient
    for index, (bias, gradient) in enumerate(zip(layer.bias, layer.grad_bias)):
        layer.bias[index] = bias - rate * gradient
    layer._has_backward = False


VARIANTS = {"A": step_a, "B": step_b, "C": step_c}
BLOCK_ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")


def _base_layer(output_size: int, input_size: int, seed: int) -> Linear:
    layer = Linear(input_size=input_size, output_size=output_size, seed=seed)
    inputs = [((index % 17) - 8) / 17.0 for index in range(input_size)]
    layer.forward(inputs)
    rng = random.Random(seed + 1)
    grad_output = [rng.uniform(-0.5, 0.5) for _ in range(output_size)]
    layer.backward(grad_output)
    return layer


def _state(layer: Linear) -> tuple[list[list[float]], list[float], bool]:
    return (
        [row.copy() for row in layer.weights],
        layer.bias.copy(),
        layer._has_backward,
    )


def _identity_result(layer: Linear, variant: str) -> dict[str, bool]:
    before_weights = layer.weights
    before_rows = layer.weights.copy()
    before_bias = layer.bias
    VARIANTS[variant](layer, 0.05)
    return {
        "weights_outer_list_preserved": layer.weights is before_weights,
        "weight_row_lists_preserved": all(
            new_row is old_row for new_row, old_row in zip(layer.weights, before_rows)
        ),
        "bias_list_preserved": layer.bias is before_bias,
        "has_backward_after_step": layer._has_backward,
    }


def _one_step_equivalence(base: Linear, learning_rate: float) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    reference_state: tuple[list[list[float]], list[float], bool] | None = None
    equality: dict[str, dict[str, bool]] = {}
    identities: dict[str, dict[str, bool]] = {}
    for name, fn in VARIANTS.items():
        layer = copy.deepcopy(base)
        fn(layer, learning_rate)
        current = _state(layer)
        if name == "A":
            reference_state = current
        assert reference_state is not None
        equality[name] = {
            "weights_exact_vs_A": current[0] == reference_state[0],
            "bias_exact_vs_A": current[1] == reference_state[1],
            "has_backward_exact_vs_A": current[2] == reference_state[2],
        }
        identities[name] = _identity_result(copy.deepcopy(base), name)
    return equality, identities


def _allocation_probe(base: Linear, name: str, learning_rate: float) -> dict[str, int]:
    layer = copy.deepcopy(base)
    gc.collect()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    VARIANTS[name](layer, learning_rate)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "live_bytes_delta": current - before_current,
        "peak_bytes_above_start": peak - before_current,
    }


def _measure(
    fn: Callable[[Linear, float], None],
    layer: Linear,
    learning_rate: float,
    iterations: int,
) -> dict[str, int]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for _ in range(iterations):
        # Same re-arm cost for all variants; each call models one valid forward/backward/step cycle.
        layer._has_backward = True
        fn(layer, learning_rate)
    return {
        "wall_ns": time.perf_counter_ns() - wall_started,
        "cpu_ns": time.process_time_ns() - cpu_started,
    }


def _benchmark_shape(output_size: int, input_size: int, iterations: int, warmups: int) -> dict[str, object]:
    base = _base_layer(output_size, input_size, seed=7101 + output_size)
    learning_rate = 0.05
    equality, identity = _one_step_equivalence(base, learning_rate)
    allocation = {
        name: _allocation_probe(base, name, learning_rate) for name in VARIANTS
    }

    warmup_order: list[str] = []
    for _ in range(warmups):
        for name, fn in VARIANTS.items():
            layer = copy.deepcopy(base)
            fn(layer, learning_rate)
            warmup_order.append(name)

    blocks: list[dict[str, object]] = []
    final_states: dict[str, list[tuple[list[list[float]], list[float], bool]]] = {
        name: [] for name in VARIANTS
    }
    for block_index, order in enumerate(BLOCK_ORDERS, start=1):
        layers = {name: copy.deepcopy(base) for name in VARIANTS}
        block: dict[str, object] = {"block": block_index, "order": list(order), "samples": {}}
        samples = block["samples"]
        assert isinstance(samples, dict)
        for name in order:
            samples[name] = _measure(VARIANTS[name], layers[name], learning_rate, iterations)
            final_states[name].append(_state(layers[name]))
        blocks.append(block)

    ratios: dict[str, dict[str, list[float]]] = {
        name: {"wall": [], "cpu": []} for name in ("B", "C")
    }
    for block in blocks:
        samples = block["samples"]
        assert isinstance(samples, dict)
        a_sample = samples["A"]
        assert isinstance(a_sample, dict)
        for name in ("B", "C"):
            sample = samples[name]
            assert isinstance(sample, dict)
            ratios[name]["wall"].append(a_sample["wall_ns"] / sample["wall_ns"])
            ratios[name]["cpu"].append(a_sample["cpu_ns"] / sample["cpu_ns"])

    block_state_equality = {
        name: [state == final_states["A"][index] for index, state in enumerate(states)]
        for name, states in final_states.items()
    }
    medians = {
        name: {
            clock: statistics.median(
                block["samples"][name][f"{clock}_ns"] / iterations  # type: ignore[index]
                for block in blocks
            )
            for clock in ("wall", "cpu")
        }
        for name in VARIANTS
    }
    paired_medians = {
        name: {clock: statistics.median(values) for clock, values in clocks.items()}
        for name, clocks in ratios.items()
    }
    paired_spread = {
        name: {
            clock: (max(values) - min(values)) / statistics.median(values) * 100
            for clock, values in clocks.items()
        }
        for name, clocks in ratios.items()
    }
    exact = all(
        all(item.values()) for item in equality.values()
    ) and all(all(values) for values in block_state_equality.values())
    wins = {
        name: {
            clock: sum(value > 1.0 for value in values)
            for clock, values in clocks.items()
        }
        for name, clocks in ratios.items()
    }
    return {
        "shape": {"output_size": output_size, "input_size": input_size},
        "learning_rate": learning_rate,
        "protocol": {
            "warmups_per_variant": warmups,
            "warmup_order": warmup_order,
            "iterations_per_sample": iterations,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "timers": ["perf_counter_ns", "process_time_ns"],
            "paired_ratio": "A_over_candidate = elapsed_A / elapsed_candidate; >1 favors candidate",
        },
        "exact_one_step_state_vs_A": equality,
        "identity_semantics_after_one_step": identity,
        "allocation_probe_tracemalloc_bytes": allocation,
        "blocks": blocks,
        "median_ns_per_step": medians,
        "paired_speedup_ratios": ratios,
        "median_paired_speedup": paired_medians,
        "paired_speedup_range_percent": paired_spread,
        "candidate_wins_out_of_six_blocks": wins,
        "exact_final_state_after_repeated_steps_by_block": block_state_equality,
        "all_numeric_and_flag_states_exact": exact,
    }


def benchmark(iterations: int, warmups: int) -> dict[str, object]:
    critical = _benchmark_shape(3932, 26, iterations, warmups)
    small = _benchmark_shape(51, 26, iterations, warmups)
    measurements = {"3932x26": critical, "51x26": small}
    exact = all(bool(shape["all_numeric_and_flag_states_exact"]) for shape in measurements.values())
    critical_speedups = critical["median_paired_speedup"]
    assert isinstance(critical_speedups, dict)
    candidate_medians = {
        name: statistics.median(
            [critical_speedups[name]["wall"], critical_speedups[name]["cpu"]]
        )
        for name in ("B", "C")
    }
    winner = max(candidate_medians, key=lambda name: candidate_medians[name])
    critical_wins = critical["candidate_wins_out_of_six_blocks"]
    small_speedups = small["median_paired_speedup"]
    small_wins = small["candidate_wins_out_of_six_blocks"]
    assert isinstance(critical_wins, dict)
    assert isinstance(small_speedups, dict)
    assert isinstance(small_wins, dict)
    decision = "review_cross_shape_regression_and_identity_contract_before_advancing"
    cross_shape_guardrail = {
        name: {
            "small_median_speedup_wall": small_speedups[name]["wall"],
            "small_median_speedup_cpu": small_speedups[name]["cpu"],
            "small_wins_out_of_six_wall": small_wins[name]["wall"],
            "small_wins_out_of_six_cpu": small_wins[name]["cpu"],
            "critical_wins_out_of_six_wall": critical_wins[name]["wall"],
            "critical_wins_out_of_six_cpu": critical_wins[name]["cpu"],
        }
        for name in ("B", "C")
    }
    return {
        "phase": "7.1_linear_step_microbenchmark",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "variant_definitions": {
            "A": "current production Linear.step()",
            "B": "float(learning_rate) once; preserve nested-list reconstruction",
            "C": "float(learning_rate) once; update weight rows and bias list in place",
        },
        "measurements": measurements,
        "exact_numeric_state_all_shapes": exact,
        "primary_shape_speedup_winner": winner,
        "primary_shape_median_speedup": candidate_medians,
        "cross_shape_guardrail": cross_shape_guardrail,
        "identity_contract_note": "C intentionally preserves outer weights/row/bias list identities; A and B replace those lists. This is observable and must be considered separately from numeric equality.",
        "production_code_changed": False,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/phase7_1_linear_step_microbenchmark.json",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmups < 0:
        parser.error("iterations deve ser positivo e warmups não pode ser negativo")
    report = benchmark(args.iterations, args.warmups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
