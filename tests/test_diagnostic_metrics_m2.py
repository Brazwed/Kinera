"""Frozen legacy behavior for diagnostic summaries before M2.1 centralization."""

from __future__ import annotations

import json
import unittest

from kinera.diagnostics.orthography import _summarize as summarize_orthography
from kinera.diagnostics.syllables import (
    _coverage,
    _grouped_pair_summary,
    _representative_examples,
    _summarize_pair,
    _target_char_effects,
)
from kinera.monitor.metrics.core import (
    oriented_delta,
    summarize_deltas,
    summarize_values,
)
from scripts.legacy.phase5_3 import _aggregate as aggregate_phase53


def _paired_row(index: int, target: str, delta_loss: float) -> dict[str, object]:
    return {
        "index": index,
        "word_key": "fixture",
        "target_char": target,
        "syllable_info": "active",
        "word_position": "middle",
        "boundary_proximity": "inside_next_syllable",
        "syllable_frequency": "rare",
        "loss": delta_loss,
        "correct": True,
        "prob_correct": 0.25,
    }


def _legacy_fields(summary: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in summary.items() if name != "metric_results"}


class DiagnosticMetricsLegacyTests(unittest.TestCase):
    def test_shared_observation_summaries_keep_example_support(self) -> None:
        self.assertEqual(
            oriented_delta(0.7, 0.2, orientation="base - candidate"),
            0.49999999999999994,
        )
        accuracy_delta = oriented_delta(
            0, 1, orientation="candidate - base"
        )
        self.assertEqual(accuracy_delta, 1)
        self.assertIs(type(accuracy_delta), int)
        self.assertEqual(summarize_values([0.3, -0.2, 0.0]), {
            "count": 3,
            "mean": 0.033333333333333326,
        })
        self.assertEqual(summarize_values([]), {"count": 0, "mean": None})
        self.assertEqual(summarize_deltas([0.3, -0.2, 0.0]), {
            "count": 3,
            "mean": 0.033333333333333326,
            "positive_count": 1,
            "negative_count": 1,
            "positive_fraction": 1 / 3,
            "negative_fraction": 1 / 3,
        })
        self.assertEqual(summarize_deltas([]), {
            "count": 0,
            "mean": None,
            "positive_count": 0,
            "negative_count": 0,
            "positive_fraction": None,
            "negative_fraction": None,
        })

    def test_syllable_pair_summary_legacy_values_and_empty_policy(self) -> None:
        rows = [
            {"delta_loss": 0.3, "delta_accuracy": 1},
            {"delta_loss": -0.2, "delta_accuracy": -1},
            {"delta_loss": 0.0, "delta_accuracy": 0},
        ]
        summary = _summarize_pair(rows)
        self.assertEqual(
            _legacy_fields(summary),
            {
                "examples": 3,
                "mean_delta_loss": 0.033333333333333326,
                "accuracy_delta": 0.0,
                "improved_loss_fraction": 1 / 3,
                "regressed_loss_fraction": 1 / 3,
            },
        )
        self.assertEqual(summary["metric_results"]["mean_delta_loss"], {
            "value": 0.033333333333333326,
            "availability": "available",
            "reason": None,
            "support": {"examples": 3},
        })
        self.assertEqual(json.loads(json.dumps(summary)), summary)
        self.assertEqual(
            _legacy_fields(_summarize_pair([])),
            {"examples": 0, "mean_delta_loss": 0.0, "accuracy_delta": 0.0},
        )
        self.assertEqual(
            _summarize_pair([])["metric_results"]["mean_delta_loss"],
            {
                "value": None,
                "availability": "not_applicable",
                "reason": "empty_support",
                "support": {"examples": 0},
            },
        )

    def test_syllable_group_order_coverage_and_ranking_ties(self) -> None:
        grouped = _grouped_pair_summary(
            [
                {"bucket": "z", "delta_loss": 2.0, "delta_accuracy": 1},
                {"bucket": "a", "delta_loss": 1.0, "delta_accuracy": 0},
                {"bucket": "z", "delta_loss": -1.0, "delta_accuracy": -1},
            ],
            "bucket",
        )
        self.assertEqual(list(grouped), ["a", "z"])
        self.assertEqual(grouped["z"]["mean_delta_loss"], 0.5)
        overall = _summarize_pair([
            {"delta_loss": 2.0, "delta_accuracy": 1},
            {"delta_loss": 1.0, "delta_accuracy": 0},
            {"delta_loss": -1.0, "delta_accuracy": -1},
        ])
        self.assertEqual(overall["mean_delta_loss"], 2 / 3)
        self.assertNotEqual(overall["mean_delta_loss"], (1.0 + 0.5) / 2)
        empty_coverage = _coverage([])
        self.assertEqual(_legacy_fields(empty_coverage), {
            "examples": 0,
            "active": 0,
            "unavailable": 0,
            "active_fraction": 0.0,
        })
        self.assertEqual(empty_coverage["metric_results"]["active_fraction"]["value"], None)
        coverage = _coverage([
            {"syllable_info": "active"},
            {"syllable_info": "unavailable"},
            {"syllable_info": "active"},
        ])
        self.assertEqual(_legacy_fields(coverage), {
            "examples": 3,
            "active": 2,
            "unavailable": 1,
            "active_fraction": 2 / 3,
        })
        self.assertEqual(coverage["metric_results"]["active_fraction"]["support"], {
            "examples": 3,
            "active": 2,
        })

        variants = {
            "A": [_paired_row(0, "b", 1.0), _paired_row(1, "a", 1.0), _paired_row(2, "c", -1.0)],
            "C": [_paired_row(0, "b", 0.0), _paired_row(1, "a", 0.0), _paired_row(2, "c", 0.0)],
            "B": [_paired_row(0, "b", 1.0), _paired_row(1, "a", 1.0), _paired_row(2, "c", -1.0)],
            "D": [_paired_row(0, "b", 0.0), _paired_row(1, "a", 0.0), _paired_row(2, "c", 0.0)],
        }
        effects = _target_char_effects(variants)["A_vs_C"]
        self.assertEqual([row["target_char"] for row in effects["most_helped"]], ["b", "a", "c"])
        self.assertEqual([row["target_char"] for row in effects["most_hurt"]], ["c", "b", "a"])

        examples = _representative_examples(variants, 3)["A_vs_C"]
        self.assertEqual([row["index"] for row in examples["gain"]], [0, 1, 2])
        self.assertEqual([row["index"] for row in examples["regression"]], [2, 0, 1])

    def test_orthographic_summary_legacy_values_and_empty_policy(self) -> None:
        rows = [
            {"delta_loss": 0.3, "delta_accuracy": 1, "base_prob_correct": 0.1, "augmented_prob_correct": 0.2},
            {"delta_loss": -0.2, "delta_accuracy": -1, "base_prob_correct": 0.2, "augmented_prob_correct": 0.3},
            {"delta_loss": 0.0, "delta_accuracy": 0, "base_prob_correct": 0.3, "augmented_prob_correct": 0.4},
        ]
        summary = summarize_orthography(rows)
        self.assertEqual(
            _legacy_fields(summary),
            {
                "count": 3,
                "mean_delta_loss": 0.033333333333333326,
                "delta_accuracy": 0.0,
                "mean_base_prob_correct": 0.19999999999999998,
                "mean_augmented_prob_correct": 0.3,
                "improved_loss_fraction": 1 / 3,
                "regressed_loss_fraction": 1 / 3,
            },
        )
        empty = summarize_orthography([])
        self.assertEqual(_legacy_fields(empty), {"count": 0})
        self.assertTrue(all(
            item["availability"] == "not_applicable" and item["value"] is None
            for item in empty["metric_results"].values()
        ))
        one = summarize_orthography([rows[0]])
        self.assertEqual(one["mean_delta_loss"], 0.3)
        self.assertEqual(one["delta_accuracy"], 1.0)

    def test_phase53_aggregate_keeps_equal_weight_seed_population(self) -> None:
        def seed(seed_id: int, loss: float, accuracy: float) -> dict[str, object]:
            return {
                "seed": seed_id,
                "status": "completed",
                "run": {
                    "pair_deltas": {
                        pair: {
                            "overall": {
                                "mean_delta_loss": loss,
                                "accuracy_delta": accuracy,
                                "improved_loss_fraction": 0.25,
                                "regressed_loss_fraction": 0.5,
                            }
                        }
                        for pair in ("A_vs_C", "B_vs_D")
                    }
                },
            }

        report = aggregate_phase53([seed(7, 0.1, 0.2), seed(42, -0.2, -0.4)])
        aggregate = report["A_vs_C"]
        self.assertEqual(aggregate["mean_delta_loss"], sum([0.1, -0.2]) / 2)
        self.assertEqual(aggregate["accuracy_delta"], sum([0.2, -0.4]) / 2)
        self.assertEqual(aggregate["metric_results"]["mean_delta_loss"]["support"], {
            "seeds": 2,
        })


if __name__ == "__main__":
    unittest.main()
