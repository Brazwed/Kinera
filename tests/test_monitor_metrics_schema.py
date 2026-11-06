"""Direct validation tests for the versioned metric result contract."""

from __future__ import annotations

import math
import unittest

from kinera.monitor.metrics.schema import metric_result


class MetricResultSchemaTests(unittest.TestCase):
    def test_available_result_is_json_native(self) -> None:
        self.assertEqual(
            metric_result(
                0.75,
                availability="available",
                reason=None,
                support={"examples": 4},
            ),
            {
                "value": 0.75,
                "availability": "available",
                "reason": None,
                "support": {"examples": 4},
            },
        )

    def test_not_applicable_and_unavailable_require_missing_values_and_reasons(self) -> None:
        for availability in ("not_applicable", "unavailable"):
            with self.subTest(availability=availability):
                result = metric_result(
                    None,
                    availability=availability,  # type: ignore[arg-type]
                    reason="support_missing",
                    support={"examples": 0},
                )
                self.assertIsNone(result["value"])
                self.assertEqual(result["availability"], availability)

                with self.assertRaises(ValueError):
                    metric_result(
                        0.0,
                        availability=availability,  # type: ignore[arg-type]
                        reason="support_missing",
                        support={"examples": 0},
                    )
                with self.assertRaises(ValueError):
                    metric_result(
                        None,
                        availability=availability,  # type: ignore[arg-type]
                        reason=None,
                        support={"examples": 0},
                    )

    def test_available_result_rejects_non_finite_values(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metric_result(
                    value,
                    availability="available",
                    reason=None,
                    support={"examples": 1},
                )

    def test_support_rejects_negative_and_boolean_counts(self) -> None:
        for support in ({"examples": -1}, {"examples": True}):
            with self.subTest(support=support), self.assertRaises(ValueError):
                metric_result(
                    1.0,
                    availability="available",
                    reason=None,
                    support=support,  # type: ignore[arg-type]
                )

    def test_available_result_rejects_boolean_values_and_reasons(self) -> None:
        with self.assertRaises(ValueError):
            metric_result(
                None,
                availability="available",
                reason=None,
                support={"examples": 1},
            )
        with self.assertRaises(TypeError):
            metric_result(
                True,
                availability="available",
                reason=None,
                support={"examples": 1},
            )
        with self.assertRaises(ValueError):
            metric_result(
                1.0,
                availability="available",
                reason="unexpected_reason",
                support={"examples": 1},
            )

    def test_invalid_availability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metric_result(
                None,
                availability="missing",  # type: ignore[arg-type]
                reason="unknown_status",
                support={"examples": 0},
            )


if __name__ == "__main__":
    unittest.main()
