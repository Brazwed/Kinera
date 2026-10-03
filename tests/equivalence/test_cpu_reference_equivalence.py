import hashlib
import json
import math
import unittest
from pathlib import Path

from kinera.experiments.comparison import run_controlled_experiments
from kinera.language.vocabulary import CharacterVocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CpuReferenceEquivalenceTests(unittest.TestCase):
    def test_frozen_cpu_oracle_artifacts_keep_their_baseline_hashes(self) -> None:
        baseline = json.loads(
            (PROJECT_ROOT / "experiments" / "phase8_0_refactor_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        for fixture in baseline["cpu_oracle_fixtures"]:
            path = PROJECT_ROOT / fixture["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.subTest(path=fixture["path"]):
                self.assertEqual(digest, fixture["sha256"])

    def test_a_b_c_d_fixture_metrics_and_parameter_counts_match_cpu_oracle(self) -> None:
        fixture = json.loads(
            (PROJECT_ROOT / "experiments" / "phase7_2_final_validation.json").read_text(
                encoding="utf-8"
            )
        )
        comparison_fixture = fixture["comparison"]
        expected_metrics = comparison_fixture["metric_snapshot"]
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        example = {
            "context_ids": [0, 3],
            "context_syllable_ids": [0, 4],
            "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
            "target_char_id": 4,
            "target_syllable_id": 4,
            "target_is_syllable_end": 1,
        }

        report = run_controlled_experiments(
            [example],
            [example],
            vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
            workers=1,
            variants=tuple(comparison_fixture["variants"]),
        )

        for variant, expected in expected_metrics.items():
            run = report["runs"][variant][0]
            actual = {
                "initial_char_loss": run["initial_validation"]["char_loss"],
                "initial_total_loss": run["initial_validation"]["total_loss"],
                "final_char_loss": run["final_validation"]["char_loss"],
                "final_total_loss": run["final_validation"]["total_loss"],
            }
            for metric_name, expected_value in expected.items():
                with self.subTest(variant=variant, metric=metric_name):
                    self.assertTrue(
                        math.isclose(actual[metric_name], expected_value, rel_tol=0.0, abs_tol=1e-15),
                        f"{variant}.{metric_name}: {actual[metric_name]} != {expected_value}",
                    )
            self.assertEqual(
                run["parameter_count"], comparison_fixture["parameter_counts"][variant]
            )


if __name__ == "__main__":
    unittest.main()
