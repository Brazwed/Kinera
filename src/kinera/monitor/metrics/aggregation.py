"""Population statistics, robust aggregation, and paired deltas for Kinera."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


def mean_std(values: list[float]) -> dict[str, float]:
    """Return population mean and standard deviation using the legacy formula."""
    if not values:
        raise ValueError("values não pode ser vazio")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


@dataclass(frozen=True)
class MetricAggregateSummary:
    """Resumo estatístico populacional com auditoria explícita de valores ausentes e inválidos.

    Invariante:
        n_values + n_missing + n_invalid == total de observações avaliadas.
    """

    n_values: int
    n_missing: int
    n_invalid: int
    seeds_used: list[int] = field(default_factory=list)
    mean: float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        if self.n_values < 0 or self.n_missing < 0 or self.n_invalid < 0:
            raise ValueError("Contadores não podem ser negativos")
        if self.n_values != len(self.seeds_used):
            raise ValueError(
                f"n_values ({self.n_values}) deve ser igual a len(seeds_used) ({len(self.seeds_used)})"
            )
        if self.n_values > 0 and (self.mean is None or self.std is None):
            raise ValueError("mean e std devem ser float quando n_values > 0")
        if self.n_values == 0 and (self.mean is not None or self.std is not None):
            raise ValueError("mean e std devem ser None quando n_values == 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_values": self.n_values,
            "n_missing": self.n_missing,
            "n_invalid": self.n_invalid,
            "seeds_used": list(self.seeds_used),
            "mean": self.mean,
            "std": self.std,
        }


def aggregate_metric_values(
    observations: Sequence[tuple[int, Any]],
) -> MetricAggregateSummary:
    """Agrega observações de uma métrica classificando valores válidos, ausentes e inválidos.

    Classificação mutuamente exclusiva para cada observação (seed, valor):
    1. Ausente (missing): valor é None (indisponibilidade científica / sem suporte).
    2. Inválido (invalid): valor não é None, mas não é número real finito (ex.: bool, str, NaN, Inf).
    3. Válido (values): valor é número real finito (int ou float, excluindo bool).

    Invariante estrita:
        n_values + n_missing + n_invalid == len(observations)
    """
    valid_values: list[float] = []
    seeds_used: list[int] = []
    n_missing = 0
    n_invalid = 0

    for seed, val in observations:
        if val is None:
            n_missing += 1
        elif isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            n_invalid += 1
        else:
            valid_values.append(float(val))
            seeds_used.append(seed)

    n_values = len(valid_values)
    if n_values > 0:
        stats = mean_std(valid_values)
        mean_val: float | None = stats["mean"]
        std_val: float | None = stats["std"]
    else:
        mean_val = None
        std_val = None

    return MetricAggregateSummary(
        n_values=n_values,
        n_missing=n_missing,
        n_invalid=n_invalid,
        seeds_used=seeds_used,
        mean=mean_val,
        std=std_val,
    )


@dataclass(frozen=True)
class PairedDeltaSummary:
    """Resumo de deltas pareados entre duas variantes com auditoria mutuamente exclusiva.

    Classificação mutuamente exclusiva por par de sementes comuns:
    - missing_pairs: pelo menos um dos valores da seed está ausente (None);
    - invalid_pairs: nenhum está ausente, mas pelo menos um é inválido (não finito ou tipo incorreto);
    - pairs: ambos são válidos e finitos.

    Invariantes:
        n_pairs == len(seeds_used)
        n_pairs + n_missing_pairs + n_invalid_pairs == n_common_seeds
    """

    orientation: str
    unit: str
    n_pairs: int
    n_missing_pairs: int
    n_invalid_pairs: int
    seeds_used: list[int] = field(default_factory=list)
    mean: float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        if self.n_pairs < 0 or self.n_missing_pairs < 0 or self.n_invalid_pairs < 0:
            raise ValueError("Contadores de pares não podem ser negativos")
        if self.n_pairs != len(self.seeds_used):
            raise ValueError(
                f"n_pairs ({self.n_pairs}) deve ser igual a len(seeds_used) ({len(self.seeds_used)})"
            )
        if self.n_pairs > 0 and (self.mean is None or self.std is None):
            raise ValueError("mean e std devem ser float quando n_pairs > 0")
        if self.n_pairs == 0 and (self.mean is not None or self.std is not None):
            raise ValueError("mean e std devem ser None quando n_pairs == 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "orientation": self.orientation,
            "unit": self.unit,
            "n_pairs": self.n_pairs,
            "n_missing_pairs": self.n_missing_pairs,
            "n_invalid_pairs": self.n_invalid_pairs,
            "seeds_used": list(self.seeds_used),
            "mean": self.mean,
            "std": self.std,
        }


def summarize_paired_deltas(
    base_observations: Mapping[int, Any],
    candidate_observations: Mapping[int, Any],
    *,
    orientation: str = "candidate - base",
    unit: str = "fraction",
) -> PairedDeltaSummary:
    """Calcula deltas pareados exclusivamente sobre seeds comuns concluídas.

    Orientação:
    - Para perdas (menor é melhor): orientation="base - candidate" (delta > 0 indica melhora do candidato)
    - Para acurácia/qualidade (maior é melhor): orientation="candidate - base" (delta > 0 indica melhora do candidato)

    Classificação mutuamente exclusiva por seed comum:
    1. missing_pairs: base ou candidate é None.
    2. invalid_pairs: nenhum é None, mas pelo menos um não é numérico finito real.
    3. pairs: ambos são válidos e finitos (calcula-se o delta orientado).

    Invariante:
        n_pairs + n_missing_pairs + n_invalid_pairs == len(common_seeds)
        n_pairs == len(seeds_used)
    """
    if orientation not in ("base - candidate", "candidate - base"):
        raise ValueError(
            f"orientation desconhecida: {orientation!r} (esperado 'base - candidate' ou 'candidate - base')"
        )

    common_seeds = sorted(set(base_observations.keys()) & set(candidate_observations.keys()))
    valid_deltas: list[float] = []
    seeds_used: list[int] = []
    n_missing_pairs = 0
    n_invalid_pairs = 0

    for seed in common_seeds:
        v_base = base_observations[seed]
        v_cand = candidate_observations[seed]

        # 1. missing_pairs: pelo menos um é ausente
        if v_base is None or v_cand is None:
            n_missing_pairs += 1
            continue

        # 2. invalid_pairs: nenhum é None, mas pelo menos um é inválido
        base_invalid = isinstance(v_base, bool) or not isinstance(v_base, (int, float)) or not math.isfinite(v_base)
        cand_invalid = isinstance(v_cand, bool) or not isinstance(v_cand, (int, float)) or not math.isfinite(v_cand)
        if base_invalid or cand_invalid:
            n_invalid_pairs += 1
            continue

        # 3. pairs: ambos são válidos e finitos
        f_base = float(v_base)
        f_cand = float(v_cand)
        if orientation == "base - candidate":
            delta = f_base - f_cand
        else:
            delta = f_cand - f_base

        valid_deltas.append(delta)
        seeds_used.append(seed)

    n_pairs = len(valid_deltas)
    if n_pairs > 0:
        stats = mean_std(valid_deltas)
        mean_val: float | None = stats["mean"]
        std_val: float | None = stats["std"]
    else:
        mean_val = None
        std_val = None

    return PairedDeltaSummary(
        orientation=orientation,
        unit=unit,
        n_pairs=n_pairs,
        n_missing_pairs=n_missing_pairs,
        n_invalid_pairs=n_invalid_pairs,
        seeds_used=seeds_used,
        mean=mean_val,
        std=std_val,
    )
