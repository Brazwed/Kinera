"""Versioned, JSON-safe descriptions for scientific metric observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal


METRICS_SCHEMA_VERSION = 1
MetricAvailability = Literal["available", "not_applicable", "unavailable"]


def metric_result(
    value: int | float | None,
    *,
    availability: MetricAvailability,
    reason: str | None,
    support: Mapping[str, int],
) -> dict[str, Any]:
    """Build one validated, JSON-native metric descriptor.

    ``support`` names the observation counts used by the metric. An
    unavailable metric has no value and explains which input is absent;
    a not-applicable metric has no value because its mathematical support is
    empty. Available values must be finite numbers.
    """
    if availability not in {"available", "not_applicable", "unavailable"}:
        raise ValueError("availability inválida")
    if availability == "available":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("metric value disponível deve ser numérico")
        if not math.isfinite(value):
            raise ValueError("metric value deve ser finito")
        if reason is not None:
            raise ValueError("metric disponível não deve ter reason")
    else:
        if value is not None:
            raise ValueError("metric indisponível não deve ter value")
        if not isinstance(reason, str) or not reason:
            raise ValueError("metric indisponível deve informar reason")

    clean_support: dict[str, int] = {}
    for name, count in support.items():
        if not isinstance(name, str) or not name:
            raise ValueError("nomes de support devem ser strings não vazias")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("contagens de support devem ser inteiros não negativos")
        clean_support[name] = count
    return {
        "value": value,
        "availability": availability,
        "reason": reason,
        "support": clean_support,
    }
