"""API M0 de métricas científicas, pura e sem efeitos colaterais.

Superfície pública:
- ``mean_std(values)`` calcula média aritmética e desvio-padrão populacional
  pela fórmula histórica do Kinera.
- ``safe_ratio(numerator, denominator, zero_denominator=...)`` calcula uma
  razão e retorna o fallback explícito (0.0 por padrão) quando o denominador é
  zero.
- ``mean_from_total(total, count)`` retorna ``total / count`` sem fallback;
  preserva ``ZeroDivisionError`` se count for zero.

O pacote usa apenas Python, não controla treino/backend e não coleta telemetry.
A atualização global de ``docs/architecture.md`` fica para o gate de integração
após a #32, para evitar conflito com trabalho concorrente.
"""

from kinera.monitor.metrics.aggregation import mean_std
from kinera.monitor.metrics.core import mean_from_total, safe_ratio

__all__ = ["mean_std", "mean_from_total", "safe_ratio"]
