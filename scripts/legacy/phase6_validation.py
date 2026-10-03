"""Valida os critérios finais da Fase 6."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples


def snapshot(model: AutoregressiveModel) -> dict[str, Any]:
    return {
        "char_embedding": [row.copy() for row in model.representation.char_embedding.weights],
        "syllable_embedding": [row.copy() for row in model.representation.syllable_embedding.weights],
        "char_head": [row.copy() for row in model.heads.char.weights],
        "syllable_head": [row.copy() for row in model.heads.syllable.weights],
        "boundary_head": [row.copy() for row in model.heads.boundary.weights],
        "char_bias": model.heads.char.bias.copy(),
        "syllable_bias": model.heads.syllable.bias.copy(),
        "boundary_bias": model.heads.boundary.bias.copy(),
    }


def examples() -> tuple[CharacterVocabulary, list[dict[str, Any]]]:
    vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
    values = [
        (0, 3, 4, 1),
        (3, 4, 3, 1),
        (4, 3, 4, 1),
    ]
    data = []
    for first, second, target, boundary in values:
        data.append({
            "context_ids": [first, second],
            "context_syllable_ids": [0, 1],
            "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
            "target_char_id": target,
            "target_syllable_id": 1,
            "target_is_syllable_end": boundary,
        })
    return vocabulary, data


def run_unittest_command(arguments: list[str]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [sys.executable, "-m", "unittest", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = f"{process.stdout}\n{process.stderr}"
    match = re.search(r"Ran (\d+) tests?", output)
    return {
        "command": arguments,
        "returncode": process.returncode,
        "tests_run": 0 if match is None else int(match.group(1)),
        "passed": process.returncode == 0,
        "output_tail": output[-2000:],
    }


def main() -> None:
    vocabulary, data = examples()
    model = AutoregressiveModel(len(vocabulary), 3, char_embedding_dim=2, syllable_embedding_dim=2, seed=7)
    initial = evaluate_multitask_examples(model, data, vocabulary)
    train_report = train_multitask_examples(model, data, vocabulary, epochs=120, learning_rate=0.2)
    final = evaluate_multitask_examples(model, data, vocabulary)
    after_training = snapshot(model)

    eval_model = AutoregressiveModel(len(vocabulary), 3, char_embedding_dim=2, syllable_embedding_dim=2, seed=7)
    before_eval = snapshot(eval_model)
    evaluation = evaluate_multitask_examples(eval_model, data, vocabulary)
    after_eval = snapshot(eval_model)

    reproducible_a = train_multitask_examples(
        AutoregressiveModel(len(vocabulary), 3, char_embedding_dim=2, syllable_embedding_dim=2, seed=7),
        data,
        vocabulary,
        epochs=4,
        learning_rate=0.1,
    )
    reproducible_b = train_multitask_examples(
        AutoregressiveModel(len(vocabulary), 3, char_embedding_dim=2, syllable_embedding_dim=2, seed=7),
        data,
        vocabulary,
        epochs=4,
        learning_rate=0.1,
    )

    full_test_run = run_unittest_command(["discover", "-s", "tests"])
    gradient_test_run = run_unittest_command([
        "tests.test_gradient_checking",
        "tests.test_multitask_loss.MultiTaskGradientCheckingTests",
        "tests.test_hybrid_integration.HybridGradientTests",
        "tests.test_linear.LinearGradientCheckingTests",
    ])

    phase51 = json.loads(Path("experiments/phase5_1_expanded_comparison.json").read_text(encoding="utf-8"))
    phase53 = json.loads(Path("experiments/phase5_3_expanded_analysis.json").read_text(encoding="utf-8"))
    report = {
        "phase": "6_final_validation",
        "criteria": {
            "all_tests_passed": full_test_run["passed"] and full_test_run["tests_run"] > 0,
            "small_subset_overfit": final["total_loss"] < initial["total_loss"],
            "validation_does_not_change_parameters": before_eval == after_eval,
            "same_seed_reproducible": reproducible_a == reproducible_b,
            "gradient_checks": gradient_test_run["passed"] and gradient_test_run["tests_run"] > 0,
            "hybrid_loss_reduction_evidence": {
                "C_vs_A_char_loss": phase51["aggregates"]["C"]["mean"]["char_loss"] < phase51["aggregates"]["A"]["mean"]["char_loss"],
                "D_vs_B_char_loss": phase51["aggregates"]["D"]["mean"]["char_loss"] < phase51["aggregates"]["B"]["mean"]["char_loss"],
                "A_vs_C_mean_delta_loss": phase53["aggregates"]["A_vs_C"]["mean_delta_loss"],
                "B_vs_D_mean_delta_loss": phase53["aggregates"]["B_vs_D"]["mean_delta_loss"],
            },
        },
        "micro_overfit": {
            "initial_total_loss": initial["total_loss"],
            "final_total_loss": final["total_loss"],
            "final_char_accuracy": final["char_accuracy"],
            "epochs": train_report["epochs"],
        },
        "evaluation_snapshot_unchanged": before_eval == after_eval,
        "trained_model_snapshot_size": len(after_training),
        "test_execution": full_test_run,
        "gradient_check_execution": gradient_test_run,
        "notes": [
            "The loss-reduction criterion is supported by held-out Phase 5.1/5.3 evidence, not by training loss alone.",
            "Phase 6.1 executes and records the full test suite and gradient checks; the issue can be closed after this validation.",
        ],
    }
    Path("experiments/phase6_final_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
