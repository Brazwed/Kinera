"""Benchmark validated internal loss helpers without editing production code."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn.losses as losses_module  # noqa: E402
import nn.probability as probability_module  # noqa: E402
from nn.losses import MultiTaskLoss  # noqa: E402


PUBLIC_API_BINDINGS = {
    "probability.softmax": probability_module.softmax,
    "probability.cross_entropy": probability_module.cross_entropy,
    "probability.cross_entropy_gradient": probability_module.cross_entropy_gradient,
    "losses.cross_entropy": losses_module.cross_entropy,
    "losses.cross_entropy_gradient": losses_module.cross_entropy_gradient,
}


def _softmax_validated(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _cross_entropy_validated(logits: list[float], target_id: int) -> float:
    maximum = max(logits)
    log_sum_exp = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return log_sum_exp - logits[target_id]


def _cross_entropy_gradient_validated(
    logits: list[float],
    target_id: int,
) -> list[float]:
    probabilities = _softmax_validated(logits)
    probabilities[target_id] -= 1.0
    return probabilities


class ValidatedHelpersLoss(MultiTaskLoss):
    """Harness-only candidate: validate once, then call arithmetic helpers."""

    def forward(
        self,
        outputs: Mapping[str, list[float]],
        targets: Mapping[str, object],
    ) -> dict[str, float]:
        self._validate_outputs(outputs)
        char_target = losses_module._validate_class_target(
            targets.get("char"), len(outputs["char"]), "char"
        )
        syllable_target = losses_module._validate_class_target(
            targets.get("syllable"), len(outputs["syllable"]), "syllable"
        )
        boundary_target = losses_module._validate_binary_target(
            targets.get("boundary")
        )
        char_loss = _cross_entropy_validated(outputs["char"], char_target)
        syllable_loss = _cross_entropy_validated(
            outputs["syllable"], syllable_target
        )
        boundary_loss = losses_module.binary_cross_entropy_with_logits(
            outputs["boundary"][0], boundary_target
        )
        return {
            "char_loss": char_loss,
            "syllable_loss": syllable_loss,
            "boundary_loss": boundary_loss,
            "total_loss": (
                self.weights["char"] * char_loss
                + self.weights["syllable"] * syllable_loss
                + self.weights["boundary"] * boundary_loss
            ),
        }

    def backward(
        self,
        outputs: Mapping[str, list[float]],
        targets: Mapping[str, object],
    ) -> dict[str, list[float]]:
        self._validate_outputs(outputs)
        char_target = losses_module._validate_class_target(
            targets.get("char"), len(outputs["char"]), "char"
        )
        syllable_target = losses_module._validate_class_target(
            targets.get("syllable"), len(outputs["syllable"]), "syllable"
        )
        boundary_target = losses_module._validate_binary_target(
            targets.get("boundary")
        )
        return {
            "char": [
                self.weights["char"] * value
                for value in _cross_entropy_gradient_validated(
                    outputs["char"], char_target
                )
            ],
            "syllable": [
                self.weights["syllable"] * value
                for value in _cross_entropy_gradient_validated(
                    outputs["syllable"], syllable_target
                )
            ],
            "boundary": [
                self.weights["boundary"]
                * losses_module.binary_cross_entropy_with_logits_gradient(
                    outputs["boundary"][0], boundary_target
                )
            ],
        }


def _make_outputs(char_size: int = 51, syllable_size: int = 3932) -> dict[str, list[float]]:
    return {
        "char": [((index * 37) % 101 - 50) / 13.0 for index in range(char_size)],
        "syllable": [
            ((index * 43) % 127 - 63) / 17.0 for index in range(syllable_size)
        ],
        "boundary": [-0.25],
    }


def _make_targets(char_size: int = 51, syllable_size: int = 3932) -> dict[str, Any]:
    return {
        "char": min(17, char_size - 1),
        "syllable": min(2213, syllable_size - 1),
        "boundary": 1.0,
    }


def _capture_error(call: Callable[[], object]) -> dict[str, Any]:
    try:
        call()
    except Exception as error:  # contract probe records exact public exception
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "NO_EXCEPTION", "message": ""}


def _public_api_error_probes() -> dict[str, dict[str, dict[str, Any]]]:
    api_calls: dict[str, Callable[[Any, Any], Any]] = {
        "probability.softmax": lambda logits, _target: probability_module.softmax(logits),
        "probability.cross_entropy": probability_module.cross_entropy,
        "probability.cross_entropy_gradient": probability_module.cross_entropy_gradient,
        "losses.cross_entropy": losses_module.cross_entropy,
        "losses.cross_entropy_gradient": losses_module.cross_entropy_gradient,
    }
    invalid_logits: dict[str, Any] = {
        "tuple_container": (0.0, 1.0),
        "none_container": None,
        "empty_list": [],
        "nan": [float("nan")],
        "positive_inf": [float("inf")],
        "negative_inf": [float("-inf")],
        "bool_value": [True],
        "string_value": ["1.0"],
    }
    invalid_targets: dict[str, Any] = {
        "bool_target": True,
        "string_target": "1",
        "none_target": None,
        "float_target": 1.0,
        "negative_target": -1,
        "out_of_range_target": 2,
    }
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for api_name, call in api_calls.items():
        probes: dict[str, dict[str, Any]] = {}
        for case_name, logits in invalid_logits.items():
            probes[f"logits.{case_name}"] = _capture_error(
                lambda call=call, logits=logits: call(logits, 0)
            )
        if not api_name.endswith("softmax"):
            for case_name, target in invalid_targets.items():
                probes[f"target.{case_name}"] = _capture_error(
                    lambda call=call, target=target: call([0.0, 1.0], target)
                )
        results[api_name] = probes
    return results


def _multitask_error_probes(
    loss_type: type[MultiTaskLoss],
) -> dict[str, dict[str, Any]]:
    valid_outputs: dict[str, Any] = _make_outputs()
    valid_targets = _make_targets()
    cases: dict[str, tuple[Any, Any]] = {}
    cases["outputs_none"] = (None, valid_targets)
    missing_head = copy.deepcopy(valid_outputs)
    del missing_head["boundary"]
    cases["outputs_missing_head"] = (missing_head, valid_targets)
    for head in ("char", "syllable"):
        cases[f"{head}_empty"] = (
            {**copy.deepcopy(valid_outputs), head: []},
            valid_targets,
        )
        for label, value in (
            ("nan", float("nan")),
            ("positive_inf", float("inf")),
            ("negative_inf", float("-inf")),
            ("bool", True),
            ("string", "bad"),
        ):
            changed: dict[str, Any] = copy.deepcopy(valid_outputs)
            changed[head][0] = value
            cases[f"{head}_{label}"] = (changed, valid_targets)
    cases["boundary_empty"] = (
        {**copy.deepcopy(valid_outputs), "boundary": []},
        valid_targets,
    )
    cases["boundary_wrong_length"] = (
        {**copy.deepcopy(valid_outputs), "boundary": [0.0, 1.0]},
        valid_targets,
    )
    for label, value in (
        ("nan", float("nan")),
        ("positive_inf", float("inf")),
        ("negative_inf", float("-inf")),
        ("bool", True),
        ("string", "bad"),
    ):
        changed: dict[str, Any] = copy.deepcopy(valid_outputs)
        changed["boundary"][0] = value
        cases[f"boundary_logit_{label}"] = (changed, valid_targets)

    target_cases: dict[str, tuple[str, Any]] = {}
    for head, out_of_range in (("char", 51), ("syllable", 3932)):
        for label, value in (
            ("bool", True),
            ("string", "1"),
            ("none", None),
            ("float", 1.0),
            ("negative", -1),
            ("out_of_range", out_of_range),
        ):
            target_cases[f"target_{head}_{label}"] = (head, value)
    for label, value in (
        ("bool", True),
        ("string", "1"),
        ("none", None),
        ("nan", float("nan")),
        ("inf", float("inf")),
        ("negative", -0.1),
        ("above_one", 1.1),
    ):
        target_cases[f"target_boundary_{label}"] = ("boundary", value)
    for case_name, (head, value) in target_cases.items():
        changed_targets = dict(valid_targets)
        changed_targets[head] = value
        cases[case_name] = (valid_outputs, changed_targets)

    missing_target = dict(valid_targets)
    del missing_target["char"]
    cases["target_char_missing"] = (valid_outputs, missing_target)
    cases["targets_none"] = (valid_outputs, None)

    results: dict[str, dict[str, Any]] = {}
    for case_name, (outputs, targets) in cases.items():
        for method_name in ("forward", "backward"):
            instance = loss_type()
            method = getattr(instance, method_name)
            results[f"{method_name}.{case_name}"] = _capture_error(
                lambda method=method, outputs=outputs, targets=targets: method(
                    outputs, targets
                )
            )
    return results


def _correctness_and_error_gate() -> dict[str, Any]:
    api_identities_before = {
        name: id(function) for name, function in PUBLIC_API_BINDINGS.items()
    }
    public_errors_before = _public_api_error_probes()
    outputs = _make_outputs()
    targets = _make_targets()
    outputs_before = copy.deepcopy(outputs)
    forward_a = MultiTaskLoss().forward(outputs, targets)
    forward_b = ValidatedHelpersLoss().forward(outputs, targets)
    backward_a = MultiTaskLoss().backward(outputs, targets)
    backward_b = ValidatedHelpersLoss().backward(outputs, targets)
    output_not_mutated = outputs == outputs_before

    multitask_errors_a = _multitask_error_probes(MultiTaskLoss)
    multitask_errors_b = _multitask_error_probes(ValidatedHelpersLoss)
    public_errors_after = _public_api_error_probes()
    api_identities_after = {
        name: id(function) for name, function in PUBLIC_API_BINDINGS.items()
    }

    public_errors_equal = public_errors_before == public_errors_after
    public_all_rejected = all(
        probe["type"] != "NO_EXCEPTION"
        for api_probes in public_errors_before.values()
        for probe in api_probes.values()
    )
    multitask_errors_equal = multitask_errors_a == multitask_errors_b
    multitask_all_rejected = all(
        result["type"] != "NO_EXCEPTION"
        for result in (*multitask_errors_a.values(), *multitask_errors_b.values())
    )
    api_identity_equal = api_identities_before == api_identities_after
    forward_exact = forward_a == forward_b
    backward_exact = backward_a == backward_b
    return {
        "forward_exact_A_vs_B": forward_exact,
        "backward_exact_A_vs_B": backward_exact,
        "input_outputs_unchanged": output_not_mutated,
        "public_api_error_cases": sum(
            len(probes) for probes in public_errors_before.values()
        ),
        "public_api_errors_exact_before_vs_after_candidate_definition": public_errors_equal,
        "public_api_all_invalid_inputs_rejected": public_all_rejected,
        "public_api_bindings_unchanged": api_identity_equal,
        "multitask_error_cases_per_method": len(multitask_errors_a) // 2,
        "multitask_exceptions_exact_A_vs_B": multitask_errors_equal,
        "multitask_all_invalid_inputs_rejected_by_both": multitask_all_rejected,
        "public_api_error_details": public_errors_before,
        "multitask_exception_details_A": multitask_errors_a,
        "multitask_exception_details_B": multitask_errors_b,
        "gate_passed": all(
            (
                forward_exact,
                backward_exact,
                output_not_mutated,
                public_errors_equal,
                public_all_rejected,
                api_identity_equal,
                multitask_errors_equal,
                multitask_all_rejected,
            )
        ),
        "forward_A": forward_a,
        "forward_B": forward_b,
        "backward_gradient_lengths": {
            "A": {head: len(values) for head, values in backward_a.items()},
            "B": {head: len(values) for head, values in backward_b.items()},
        },
    }


def _measure(
    loss: MultiTaskLoss,
    method_name: str,
    outputs: dict[str, list[float]],
    targets: dict[str, object],
    iterations: int,
) -> dict[str, int]:
    method = getattr(loss, method_name)
    gc.collect()
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    for _ in range(iterations):
        method(outputs, targets)
    return {
        "wall_ns": time.perf_counter_ns() - wall_start,
        "cpu_ns": time.process_time_ns() - cpu_start,
    }


def _benchmark(
    iterations: int,
    warmups: int,
) -> dict[str, Any]:
    outputs = _make_outputs()
    targets = _make_targets()
    classes: dict[str, type[MultiTaskLoss]] = {
        "A": MultiTaskLoss,
        "B": ValidatedHelpersLoss,
    }
    methods = ("forward", "backward")
    warmup_order: list[dict[str, str]] = []
    for _ in range(warmups):
        for variant in ("A", "B"):
            instance = classes[variant]()
            for method_name in methods:
                getattr(instance, method_name)(outputs, targets)
            warmup_order.append({"variant": variant, "methods": "forward,backward"})

    block_orders = ("AB", "BA", "AB", "BA", "AB", "BA")
    instances = {variant: classes[variant]() for variant in classes}
    blocks: list[dict[str, Any]] = []
    for block_index, order in enumerate(block_orders, start=1):
        block: dict[str, Any] = {
            "block": block_index,
            "order": list(order),
            "samples": {
                variant: {} for variant in classes
            },
        }
        samples = block["samples"]
        for variant in order:
            for method_name in methods:
                sample = _measure(
                    instances[variant],
                    method_name,
                    outputs,
                    targets,
                    iterations,
                )
                sample["iterations"] = iterations
                samples[variant][method_name] = sample
        blocks.append(block)

    per_method: dict[str, Any] = {}
    for method_name in methods:
        ratios: dict[str, list[float]] = {"wall": [], "cpu": []}
        medians: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        wins: dict[str, int] = {"wall": 0, "cpu": 0}
        for clock in ("wall", "cpu"):
            elapsed_by_variant = {
                variant: [
                    block["samples"][variant][method_name][f"{clock}_ns"]
                    for block in blocks
                ]
                for variant in classes
            }
            for variant in classes:
                medians[variant][clock] = statistics.median(
                    value / iterations for value in elapsed_by_variant[variant]
                )
            ratios[clock] = [
                a / b
                for a, b in zip(
                    elapsed_by_variant["A"], elapsed_by_variant["B"]
                )
            ]
            wins[clock] = sum(ratio > 1.0 for ratio in ratios[clock])
        per_method[method_name] = {
            "median_ns_per_call": medians,
            "paired_speedup_ratios_A_over_B": ratios,
            "median_paired_speedup": {
                clock: statistics.median(values) for clock, values in ratios.items()
            },
            "median_absolute_deviation_of_speedup": {
                clock: statistics.median(
                    abs(value - statistics.median(values)) for value in values
                )
                for clock, values in ratios.items()
            },
            "candidate_wins_out_of_six": wins,
        }

    combined_ratios: dict[str, list[float]] = {"wall": [], "cpu": []}
    for clock in ("wall", "cpu"):
        for block in blocks:
            total: dict[str, int] = {}
            for variant in classes:
                total[variant] = sum(
                    block["samples"][variant][method_name][f"{clock}_ns"]
                    for method_name in methods
                )
            combined_ratios[clock].append(total["A"] / total["B"])
    combined_summary = {
        clock: {
            "ratios_A_over_B": values,
            "median_speedup": statistics.median(values),
            "median_absolute_deviation": statistics.median(
                abs(value - statistics.median(values)) for value in values
            ),
            "candidate_wins_out_of_six": sum(value > 1.0 for value in values),
        }
        for clock, values in combined_ratios.items()
    }
    return {
        "shape": {"char_logits": len(outputs["char"]), "syllable_logits": len(outputs["syllable"])},
        "input_generation": "deterministic finite lists; char[i]=((i*37)%101-50)/13; syllable[i]=((i*43)%127-63)/17; boundary=-0.25",
        "target_ids": {"char": targets["char"], "syllable": targets["syllable"], "boundary": targets["boundary"]},
        "protocol": {
            "warmups_per_variant": warmups,
            "warmup_method_calls_per_variant": warmups * len(methods),
            "iterations_per_sample": iterations,
            "blocks": len(blocks),
            "block_orders": [list(order) for order in block_orders],
            "timers": ["perf_counter_ns", "process_time_ns"],
            "speedup_definition": "A_elapsed / B_elapsed; values >1 favor B",
            "predeclared_advance_gate": "exact values/errors; combined median wall speedup >=1.03, >=5/6 combined wall wins, and neither operation's median wall speedup below 0.99",
        },
        "blocks": blocks,
        "methods": per_method,
        "combined_forward_plus_backward": combined_summary,
    }


def run(iterations: int, warmups: int) -> dict[str, Any]:
    correctness = _correctness_and_error_gate()
    report: dict[str, Any] = {
        "phase": "7.1_validated_loss_helpers_microbenchmark",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "candidate_definitions": {
            "A": "current MultiTaskLoss.forward/backward; _validate_outputs followed by public validated cross_entropy / cross_entropy_gradient / softmax path",
            "B": "harness-only subclass preserves output/target validation, then calls arithmetic-only internal validated helpers; public APIs are not patched",
        },
        "correctness_and_public_error_contract_gate": correctness,
        "production_code_changed": False,
        "workload_D_run": False,
    }
    if not correctness["gate_passed"]:
        report["decision"] = "reject_candidate_error_or_exactness_mismatch"
        report["timing_measurements"] = None
        return report

    timing = _benchmark(iterations, warmups)
    report["timing_measurements"] = timing
    methods = timing["methods"]
    combined_wall = timing["combined_forward_plus_backward"]["wall"]
    med_forward = methods["forward"]["median_paired_speedup"]["wall"]
    med_backward = methods["backward"]["median_paired_speedup"]["wall"]
    combined_pass = (
        combined_wall["median_speedup"] >= 1.03
        and combined_wall["candidate_wins_out_of_six"] >= 5
        and med_forward >= 0.99
        and med_backward >= 0.99
    )
    report["decision"] = (
        "advance_to_separate_full_D_trial"
        if combined_pass
        else "repeat_or_reject_microbenchmark_candidate"
    )
    report["note"] = "This harness is only a candidate filter; even a pass does not authorize production edits or substitute for a separately approved full-D gate."
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/phase7_1_validated_loss_helpers_microbenchmark.json",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmups < 0:
        parser.error("iterations deve ser positivo e warmups não pode ser negativo")
    report = run(args.iterations, args.warmups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
