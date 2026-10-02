"""Compare isolated Linear.backward() dot-product strategies without production edits."""

from __future__ import annotations

import argparse
import json
import operator
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn.linear import Linear  # noqa: E402


def _validate_like_production(layer: Linear, grad_output: list[float]) -> None:
    if layer.last_input is None or layer.output is None:
        raise RuntimeError("backward exige um forward anterior")
    if not isinstance(grad_output, list):
        raise TypeError("grad_output deve ser uma lista")
    if len(grad_output) != len(layer.weights):
        raise ValueError("grad_output deve ter o mesmo tamanho das saídas")
    for index, value in enumerate(grad_output):
        layer._validate_number(value, f"grad_output[{index}]")


def current_a(layer: Linear, grad_output: list[float]) -> list[float]:
    """A: invoke the unchanged production implementation."""
    return layer.backward(grad_output)


def converted_b(layer: Linear, grad_output: list[float]) -> list[float]:
    """B: convert grad_output once, retain generator sums and row-major weights."""
    _validate_like_production(layer, grad_output)
    assert layer.last_input is not None
    grad_output_float = [float(value) for value in grad_output]
    layer.grad_weights = [
        [grad_output_float[row_index] * input_value for input_value in layer.last_input]
        for row_index in range(len(layer.weights))
    ]
    layer.grad_bias = grad_output_float.copy()
    grad_inputs = [0.0 for _ in layer.last_input]
    for input_index in range(len(layer.last_input)):
        grad_inputs[input_index] = sum(
            grad_output_float[row_index] * layer.weights[row_index][input_index]
            for row_index in range(len(layer.weights))
        )
    layer._has_backward = True
    return grad_inputs


def transposed_map_c(layer: Linear, grad_output: list[float]) -> list[float]:
    """C: convert once, transpose weights per call, then map-multiply each column."""
    _validate_like_production(layer, grad_output)
    assert layer.last_input is not None
    grad_output_float = [float(value) for value in grad_output]
    columns = [list(column) for column in zip(*layer.weights)]
    layer.grad_weights = [
        [grad_output_float[row_index] * input_value for input_value in layer.last_input]
        for row_index in range(len(layer.weights))
    ]
    layer.grad_bias = grad_output_float.copy()
    grad_inputs: list[float] = [
        float(sum(map(operator.mul, column, grad_output_float))) for column in columns
    ]
    layer._has_backward = True
    return grad_inputs


VARIANTS: dict[str, Callable[[Linear, list[float]], list[float]]] = {
    "A": current_a,
    "B": converted_b,
    "C": transposed_map_c,
}

# Balanced six-permutation schedule: each variant occupies each position twice.
BLOCK_ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")


def _snapshot(layer: Linear, grad_inputs: list[float]) -> tuple[list[float], list[list[float]], list[float], bool]:
    return (
        grad_inputs.copy(),
        [row.copy() for row in layer.grad_weights],
        layer.grad_bias.copy(),
        layer._has_backward,
    )


def _timed_sample(
    fn: Callable[[Linear, list[float]], list[float]],
    layer: Linear,
    grad_output: list[float],
    iterations: int,
) -> dict[str, int]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for _ in range(iterations):
        fn(layer, grad_output)
    return {
        "wall_ns": time.perf_counter_ns() - wall_started,
        "cpu_ns": time.process_time_ns() - cpu_started,
    }


def benchmark(iterations: int, warmups: int) -> dict[str, object]:
    input_size, output_size = 26, 3932
    layer = Linear(input_size=input_size, output_size=output_size, seed=7101)
    inputs = [float(index + 1) / 27 for index in range(input_size)]
    layer.forward(inputs)
    rng = random.Random(7102)
    grad_output = [rng.uniform(-0.5, 0.5) for _ in range(output_size)]

    reference_inputs = current_a(layer, grad_output)
    reference = _snapshot(layer, reference_inputs)
    equality: dict[str, dict[str, bool]] = {}
    for name in ("B", "C"):
        candidate_inputs = VARIANTS[name](layer, grad_output)
        candidate = _snapshot(layer, candidate_inputs)
        equality[name] = {
            "grad_inputs_exact": candidate[0] == reference[0],
            "grad_weights_exact": candidate[1] == reference[1],
            "grad_bias_exact": candidate[2] == reference[2],
            "backward_state_exact": candidate[3] == reference[3],
        }
    equality["A"] = {
        "grad_inputs_exact": True,
        "grad_weights_exact": True,
        "grad_bias_exact": True,
        "backward_state_exact": True,
    }

    warmup_order: list[str] = []
    for _ in range(warmups):
        for name, fn in VARIANTS.items():
            fn(layer, grad_output)
            warmup_order.append(name)

    blocks: list[dict[str, object]] = []
    for block_index, order in enumerate(BLOCK_ORDERS, start=1):
        block: dict[str, object] = {"block": block_index, "order": list(order), "samples": {}}
        samples = block["samples"]
        assert isinstance(samples, dict)
        for name in order:
            samples[name] = _timed_sample(VARIANTS[name], layer, grad_output, iterations)
        blocks.append(block)

    ratios: dict[str, dict[str, list[float]]] = {"B": {"wall": [], "cpu": []}, "C": {"wall": [], "cpu": []}}
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

    transpose_started_wall = time.perf_counter_ns()
    transpose_started_cpu = time.process_time_ns()
    columns = [list(column) for column in zip(*layer.weights)]
    transpose_only = {
        "wall_ns": time.perf_counter_ns() - transpose_started_wall,
        "cpu_ns": time.process_time_ns() - transpose_started_cpu,
    }
    transpose_shallow_bytes = sys.getsizeof(columns) + sum(sys.getsizeof(column) for column in columns)

    return {
        "phase": "7.1_linear_backward_microbenchmark",
        "shape": {"input_size": input_size, "output_size": output_size},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "protocol": {
            "warmups_per_variant": warmups,
            "warmup_order": warmup_order,
            "iterations_per_sample": iterations,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "timers": ["perf_counter_ns", "process_time_ns"],
            "ratio_definition": "speedup_A_over_candidate = elapsed_A / elapsed_candidate; >1 favors candidate",
        },
        "variants": {
            "A": "current production Linear.backward()",
            "B": "one float conversion of grad_output; original row-major generator sum retained",
            "C": "one float conversion plus per-call weight transpose; map(operator.mul) on each column; transpose cost included in timed function",
        },
        "exact_equality_vs_A": equality,
        "blocks": blocks,
        "paired_speedup_ratios": ratios,
        "median_paired_speedup": {
            name: {clock: statistics.median(values) for clock, values in clocks.items()}
            for name, clocks in ratios.items()
        },
        "paired_speedup_range_percent": {
            name: {
                clock: (max(values) - min(values)) / statistics.median(values) * 100
                for clock, values in clocks.items()
            }
            for name, clocks in ratios.items()
        },
        "transpose_cost_C": {
            "single_materialization": transpose_only,
            "shallow_container_bytes": transpose_shallow_bytes,
            "note": "C's timed backward samples include transposition; this standalone sample only isolates that cost.",
        },
        "production_code_changed": False,
        "decision": "advance_candidate_to_full_benchmark_only_if_exact_and_speedup_consistent",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/phase7_1_linear_backward_microbenchmark.json",
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
