"""Baselines determinísticos de previsão para classificação binária.

Este módulo implementa heurísticas puras de referência:
- Baseline constante 0: prediz sempre 0 sobre os alvos avaliados.
- Baseline constante 1: prediz sempre 1 sobre os alvos avaliados.
- Baseline de classe majoritária: seleciona a classe mais frequente no conjunto de
  treino (com desempate determinístico para 0) e avalia essa constante sobre o conjunto
  de avaliação.

Esses baselines contextualizam as métricas observadas no modelo, evitando que pontuações
decorrentes de desbalanceamento sejam erroneamente atribuídas a aprendizado representacional.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from kinera.monitor.metrics.confusion import (
    BinaryConfusion,
    _validate_count,
    binary_classification_metrics,
)


def _validate_binary_target(value: Any, name: str = "target") -> int:
    """Valida que o valor é um inteiro 0 ou 1 (rejeita booleano)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise TypeError(f"{name} deve ser 0 ou 1 inteiro (recebeu {value!r})")
    return value


def constant_baseline_from_counts(
    support_positive: int,
    support_negative: int,
    constant_prediction: int,
) -> dict[str, Any]:
    """Calcula métricas de previsão constante diretamente a partir das contagens de alvos.

    Args:
        support_positive: Número de instâncias da classe 1 no conjunto avaliado.
        support_negative: Número de instâncias da classe 0 no conjunto avaliado.
        constant_prediction: Valor fixo previsto para todas as instâncias (0 ou 1).

    Returns:
        Dicionário com a previsão constante aplicada e as métricas resultantes.
    """
    _validate_count(support_positive, "support_positive")
    _validate_count(support_negative, "support_negative")
    pred = _validate_binary_target(constant_prediction, "constant_prediction")

    if pred == 1:
        # Prevê sempre 1: todos os positivos viram TP; todos os negativos viram FP.
        confusion = BinaryConfusion(
            tp=support_positive,
            tn=0,
            fp=support_negative,
            fn=0,
        )
    else:
        # Prevê sempre 0: todos os positivos viram FN; todos os negativos viram TN.
        confusion = BinaryConfusion(
            tp=0,
            tn=support_negative,
            fp=0,
            fn=support_positive,
        )

    metrics = binary_classification_metrics(confusion)
    metrics["constant_prediction"] = pred
    return metrics


def evaluate_constant_baseline(
    targets: Sequence[int],
    constant_prediction: int,
) -> dict[str, Any]:
    """Avalia previsão constante sobre uma sequência de alvos binários."""
    pos = 0
    neg = 0
    for idx, t in enumerate(targets):
        val = _validate_binary_target(t, f"target[{idx}]")
        if val == 1:
            pos += 1
        else:
            neg += 1
    return constant_baseline_from_counts(pos, neg, constant_prediction)


def majority_class(
    train_data: Sequence[int] | Mapping[int | str, int],
) -> int:
    """Identifica a classe majoritária do conjunto de treino com desempate determinístico para 0.

    Args:
        train_data: Sequência de alvos inteiros {0, 1} ou mapeamento com contagens
            (ex.: {"0": 453, "1": 547} ou {0: 453, 1: 547}).

    Returns:
        0 ou 1 representando a classe com maior frequência. Em caso de empate exato,
        retorna 0 deterministamente.
    """
    if isinstance(train_data, Mapping):
        c0 = 0
        c1 = 0
        for k, v in train_data.items():
            _validate_count(v, f"contagem da classe {k!r}")
            k_norm = int(k) if isinstance(k, str) and k.isdigit() else k
            if isinstance(k_norm, bool) or not isinstance(k_norm, int) or k_norm not in (0, 1):
                raise TypeError(f"Chave de classe deve ser 0 ou 1 (recebeu {k!r})")
            if k_norm == 0:
                c0 += v
            else:
                c1 += v
    elif isinstance(train_data, Sequence):
        if not train_data:
            raise ValueError("train_data não pode ser vazio para determinar classe majoritária")
        c0 = 0
        c1 = 0
        for idx, t in enumerate(train_data):
            val = _validate_binary_target(t, f"train_target[{idx}]")
            if val == 0:
                c0 += 1
            else:
                c1 += 1
    else:
        raise TypeError(
            f"train_data deve ser Sequence ou Mapping (recebeu {type(train_data).__name__})"
        )

    if c0 + c1 == 0:
        raise ValueError("Total de instâncias de treino deve ser maior que zero")

    # Desempate determinístico para 0:
    return 0 if c0 >= c1 else 1


def evaluate_majority_baseline(
    train_data: Sequence[int] | Mapping[int | str, int],
    eval_targets: Sequence[int],
) -> dict[str, Any]:
    """Avalia o baseline de classe majoritária do treino sobre os alvos de avaliação."""
    chosen = majority_class(train_data)
    result = evaluate_constant_baseline(eval_targets, chosen)
    result["majority_class"] = chosen
    return result


def evaluate_binary_baselines(
    eval_targets: Sequence[int],
    *,
    train_data: Sequence[int] | Mapping[int | str, int] | None = None,
) -> dict[str, Any]:
    """Gera o conjunto completo de baselines determinísticos para os alvos de avaliação.

    Args:
        eval_targets: Alvos binários da partição avaliada.
        train_data: Opcional; alvos ou contagens de treino para o baseline de classe majoritária.

    Returns:
        Dicionário com baselines 'constant_0', 'constant_1' e opcionalmente 'majority'.
    """
    res: dict[str, Any] = {
        "constant_0": evaluate_constant_baseline(eval_targets, 0),
        "constant_1": evaluate_constant_baseline(eval_targets, 1),
    }
    if train_data is not None:
        res["majority"] = evaluate_majority_baseline(train_data, eval_targets)
    return res
