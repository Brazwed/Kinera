"""Matriz de confusão binária e métricas científicas derivadas puras.

Este módulo implementa o núcleo matemático de avaliação de classificadores binários
(como a head de fronteira silábica), garantindo:
- Validação estrita de tipos e contadores (inteiros não-negativos, sem booleanos).
- Fórmulas canônicas para precisão, recall, especificidade, F1 e Balanced Accuracy.
- Representação explícita de valores indisponíveis por ``None`` (nunca NaN ou Infinity).
- Cálculo direto de F1 a partir dos contadores brutos, permitindo F1=0.0 com precisão indefinida.
- Execução puramente CPU-only sem dependências externas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _validate_count(value: Any, name: str) -> int:
    """Valida estritamente se um contador é um inteiro não-negativo (rejeita bools)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} deve ser um inteiro (recebeu {type(value).__name__})"
        )
    if value < 0:
        raise ValueError(f"{name} deve ser não-negativo (recebeu {value})")
    return value


@dataclass(frozen=True)
class BinaryConfusion:
    """Matriz de confusão binária 2x2 imutável."""

    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        _validate_count(self.tp, "tp")
        _validate_count(self.tn, "tn")
        _validate_count(self.fp, "fp")
        _validate_count(self.fn, "fn")

    @property
    def support_positive(self) -> int:
        """Suporte da classe positiva: TP + FN."""
        return self.tp + self.fn

    @property
    def support_negative(self) -> int:
        """Suporte da classe negativa: TN + FP."""
        return self.tn + self.fp

    @property
    def predicted_positive(self) -> int:
        """Total de previsões positivas: TP + FP."""
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> int:
        """Total de previsões negativas: TN + FN."""
        return self.tn + self.fn

    @property
    def total(self) -> int:
        """Total de instâncias avaliadas: TP + TN + FP + FN."""
        return self.tp + self.tn + self.fp + self.fn

    def __add__(self, other: object) -> BinaryConfusion:
        if not isinstance(other, BinaryConfusion):
            return NotImplemented
        return BinaryConfusion(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def to_dict(self) -> dict[str, int]:
        """Serializa os contadores brutos para dicionário."""
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinaryConfusion:
        """Reconstrói BinaryConfusion a partir de dicionário com validação."""
        if not isinstance(data, dict):
            raise TypeError("data deve ser um dicionário")
        required = {"tp", "tn", "fp", "fn"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Chaves ausentes em data: {sorted(missing)}")
        return cls(
            tp=data["tp"],
            tn=data["tn"],
            fp=data["fp"],
            fn=data["fn"],
        )

    @classmethod
    def from_sequences(
        cls,
        targets: Sequence[int],
        predictions: Sequence[int],
    ) -> BinaryConfusion:
        """Contrói BinaryConfusion a partir de sequências de targets e predictions."""
        if len(targets) != len(predictions):
            raise ValueError(
                f"targets (len={len(targets)}) e predictions (len={len(predictions)}) "
                "devem ter o mesmo comprimento"
            )
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for idx, (t, p) in enumerate(zip(targets, predictions)):
            if isinstance(t, bool) or not isinstance(t, int) or t not in (0, 1):
                raise TypeError(
                    f"target no índice {idx} deve ser 0 ou 1 inteiro (recebeu {t!r})"
                )
            if isinstance(p, bool) or not isinstance(p, int) or p not in (0, 1):
                raise TypeError(
                    f"prediction no índice {idx} deve ser 0 ou 1 inteiro (recebeu {p!r})"
                )
            if p == 1 and t == 1:
                tp += 1
            elif p == 0 and t == 0:
                tn += 1
            elif p == 1 and t == 0:
                fp += 1
            else:
                fn += 1
        return cls(tp=tp, tn=tn, fp=fp, fn=fn)


def precision(confusion: BinaryConfusion) -> float | None:
    """Precisão binária: TP / (TP + FP). Retorna None se não houver previsões positivas."""
    if not isinstance(confusion, BinaryConfusion):
        raise TypeError("confusion deve ser uma instância de BinaryConfusion")
    denominator = confusion.predicted_positive
    if denominator == 0:
        return None
    return confusion.tp / denominator


def recall(confusion: BinaryConfusion) -> float | None:
    """Recall binário: TP / (TP + FN). Retorna None se não houver suporte positivo."""
    if not isinstance(confusion, BinaryConfusion):
        raise TypeError("confusion deve ser uma instância de BinaryConfusion")
    denominator = confusion.support_positive
    if denominator == 0:
        return None
    return confusion.tp / denominator


def specificity(confusion: BinaryConfusion) -> float | None:
    """Especificidade binária: TN / (TN + FP). Retorna None se não houver suporte negativo."""
    if not isinstance(confusion, BinaryConfusion):
        raise TypeError("confusion deve ser uma instância de BinaryConfusion")
    denominator = confusion.support_negative
    if denominator == 0:
        return None
    return confusion.tn / denominator


def f1_score(confusion: BinaryConfusion) -> float | None:
    """F1-Score binário calculado diretamente dos contadores brutos: 2TP / (2TP + FP + FN).

    Retorna None se o denominador (2TP + FP + FN) for zero.
    Se o denominador for maior que zero mas TP for zero, retorna 0.0 (mesmo que
    a precisão seja indefinida).
    """
    if not isinstance(confusion, BinaryConfusion):
        raise TypeError("confusion deve ser uma instância de BinaryConfusion")
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    if denominator == 0:
        return None
    return (2.0 * confusion.tp) / denominator


def balanced_accuracy(confusion: BinaryConfusion) -> float | None:
    """Balanced Accuracy: média aritmética entre recall e especificidade.

    Retorna None se qualquer uma das duas métricas for indefinida (ou seja,
    quando uma das classes não possui suporte no conjunto avaliado).
    """
    rec = recall(confusion)
    spec = specificity(confusion)
    if rec is None or spec is None:
        return None
    return (rec + spec) / 2.0


def binary_classification_metrics(confusion: BinaryConfusion) -> dict[str, Any]:
    """Calcula e resume todas as métricas a partir de uma BinaryConfusion."""
    if not isinstance(confusion, BinaryConfusion):
        raise TypeError("confusion deve ser uma instância de BinaryConfusion")
    return {
        "confusion": confusion.to_dict(),
        "support_positive": confusion.support_positive,
        "support_negative": confusion.support_negative,
        "predicted_positive": confusion.predicted_positive,
        "predicted_negative": confusion.predicted_negative,
        "total": confusion.total,
        "precision": precision(confusion),
        "recall": recall(confusion),
        "specificity": specificity(confusion),
        "f1": f1_score(confusion),
        "balanced_accuracy": balanced_accuracy(confusion),
    }
