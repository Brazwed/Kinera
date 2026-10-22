"""Métricas científicas de segmentação silábica por offsets internos de palavras.

Este módulo implementa a avaliação pura de fronteiras de segmentação inter-silábica,
distinguindo explicitamente:
- Correspondência exata da palavra (todas as fronteiras coincidem perfeitamente).
- Avaliação por offsets de corte: TP (corte correto), FP (corte espúrio),
  FN (corte omitido), TN (posição interna corretamente não cortada).
- Universo de candidatos a corte: para palavra de comprimento L, existem exatamente
  L - 1 posições internas candidatas (1, ..., L - 1). Para L <= 1, existem 0 candidatos.
- Comportamento de palavras sem posições candidatas (L <= 1):
  Palavras vazias (L=0) ou unicaracteres (L=1) não admitem limites internos de corte
  (candidate_positions = 0). Quando ambos os conjuntos de offsets (referência e previsão)
  são vazios, a palavra é considerada uma correspondência exata válida (exact_match=True)
  e é contabilizada tanto em exact_matches quanto no denominador words_count.
  Consequentemente, palavras sem posições candidatas influenciam a taxa word_exact_match_rate,
  mas contribuem com 0 posições para as métricas de offsets (TP=TN=FP=FN=0).
- Execução puramente CPU-only sem efeitos colaterais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from kinera.monitor.metrics.confusion import _validate_count


def _normalize_offsets(
    offsets: Iterable[int],
    length: int,
    name: str = "offsets",
) -> set[int]:
    """Valida e normaliza offsets de corte para uma palavra de comprimento length.

    - Rejeita valores não-inteiros e booleanos.
    - Se length <= 1, rejeita qualquer offset fornecido (não há cortes válidos).
    - Se length > 1, valida que cada offset pertence a [1, length - 1].
    - Elimina duplicatas de forma determinística retornando um conjunto (set).
    """
    if offsets is None or isinstance(offsets, (str, bytes)):
        raise TypeError(f"{name} deve ser um iterável de inteiros (recebeu {type(offsets).__name__})")

    result: set[int] = set()
    for item in offsets:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"{name} deve conter inteiros (recebeu elemento {item!r} do tipo {type(item).__name__})"
            )
        if length <= 1:
            raise ValueError(
                f"{name} contém offset {item} para palavra de comprimento {length} <= 1 "
                "(não existem limites internos candidatos a corte)"
            )
        if item < 1 or item >= length:
            raise ValueError(
                f"{name} contém offset {item} fora dos limites válidos [1, {length - 1}] "
                f"para palavra de comprimento {length}"
            )
        result.add(item)
    return result


@dataclass(frozen=True)
class WordSegmentationResult:
    """Resultado da avaliação de segmentação para uma única palavra.

    Atributos:
        word_length: Comprimento L da palavra.
        candidate_positions: Número de posições internas candidatas (max(0, L - 1)).
        tp: Cortes corretos (Gold ∩ Pred).
        tn: Posições internas sem corte em Gold e Pred.
        fp: Cortes espúrios previstos (Pred - Gold).
        fn: Cortes omitidos (Gold - Pred).
        exact_match: Indica se todos os offsets previstos coincidem exatamente com o gabarito.
            Para palavras com L <= 1 (sem limites internos candidatos), exact_match é True
            sempre que nenhum corte espúrio for atribuído a ela.
    """

    word_length: int
    candidate_positions: int
    tp: int
    tn: int
    fp: int
    fn: int
    exact_match: bool


@dataclass(frozen=True)
class OffsetSegmentationConfusion:
    """Agregação de confusão e métricas de segmentação para um conjunto de palavras.

    Atributos:
        words_count: Total de palavras avaliadas no corpus (inclui palavras com L <= 1).
        exact_matches: Total de palavras com exata correspondência de cortes (inclui
            palavras com L <= 1 sem cortes).
        candidate_positions: Total de posições internas candidatas a corte no corpus.
        tp, tn, fp, fn: Contadores acumulados de offsets.
    """

    words_count: int
    exact_matches: int
    candidate_positions: int
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        _validate_count(self.words_count, "words_count")
        _validate_count(self.exact_matches, "exact_matches")
        _validate_count(self.candidate_positions, "candidate_positions")
        _validate_count(self.tp, "tp")
        _validate_count(self.tn, "tn")
        _validate_count(self.fp, "fp")
        _validate_count(self.fn, "fn")
        if self.exact_matches > self.words_count:
            raise ValueError(
                f"exact_matches ({self.exact_matches}) não pode exceder words_count ({self.words_count})"
            )
        if self.tp + self.tn + self.fp + self.fn != self.candidate_positions:
            raise ValueError(
                f"Soma dos contadores ({self.tp} + {self.tn} + {self.fp} + {self.fn} = "
                f"{self.tp + self.tn + self.fp + self.fn}) deve ser igual a "
                f"candidate_positions ({self.candidate_positions})"
            )

    @property
    def support_positive(self) -> int:
        """Total de cortes gold existentes: TP + FN."""
        return self.tp + self.fn

    @property
    def support_negative(self) -> int:
        """Total de posições internas de não-corte: TN + FP."""
        return self.tn + self.fp

    @property
    def predicted_positive(self) -> int:
        """Total de cortes previstos pelo modelo: TP + FP."""
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> int:
        """Total de posições internas não cortadas pelo modelo: TN + FN."""
        return self.tn + self.fn

    @property
    def total_positions(self) -> int:
        """Total de posições internas avaliadas."""
        return self.candidate_positions

    def __add__(self, other: object) -> OffsetSegmentationConfusion:
        if not isinstance(other, OffsetSegmentationConfusion):
            return NotImplemented
        return OffsetSegmentationConfusion(
            words_count=self.words_count + other.words_count,
            exact_matches=self.exact_matches + other.exact_matches,
            candidate_positions=self.candidate_positions + other.candidate_positions,
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def to_dict(self) -> dict[str, int]:
        """Serializa os contadores brutos para dicionário."""
        return {
            "words_count": self.words_count,
            "exact_matches": self.exact_matches,
            "candidate_positions": self.candidate_positions,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OffsetSegmentationConfusion:
        """Reconstrói OffsetSegmentationConfusion a partir de dicionário com validação."""
        if not isinstance(data, dict):
            raise TypeError("data deve ser um dicionário")
        required = {
            "words_count",
            "exact_matches",
            "candidate_positions",
            "tp",
            "tn",
            "fp",
            "fn",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Chaves ausentes em data: {sorted(missing)}")
        return cls(
            words_count=data["words_count"],
            exact_matches=data["exact_matches"],
            candidate_positions=data["candidate_positions"],
            tp=data["tp"],
            tn=data["tn"],
            fp=data["fp"],
            fn=data["fn"],
        )


def evaluate_word_offsets(
    word_or_length: str | int,
    gold_offsets: Iterable[int],
    pred_offsets: Iterable[int],
) -> WordSegmentationResult:
    """Avalia o corte inter-silábico para uma única palavra.

    Args:
        word_or_length: A palavra (string) ou seu comprimento inteiro L >= 0.
        gold_offsets: Conjunto ou iterável de offsets de corte de referência.
        pred_offsets: Conjunto ou iterável de offsets de corte previstos.

    Returns:
        WordSegmentationResult com contagens de TP, TN, FP, FN e exact_match.

    Nota sobre palavras sem candidatos (L <= 1):
        Para palavras com comprimento L <= 1 (vazias ou de 1 caractere), não existem
        limites internos candidatos (candidate_positions = 0). Se nem a referência nem
        a previsão possuírem offsets, exact_match será True, e a palavra contabilizará
        1 em words_count e 1 em exact_matches quando agregada no corpus.
    """
    if isinstance(word_or_length, str):
        length = len(word_or_length)
    elif isinstance(word_or_length, bool) or not isinstance(word_or_length, int):
        raise TypeError(
            f"word_or_length deve ser str ou int (recebeu {type(word_or_length).__name__})"
        )
    else:
        if word_or_length < 0:
            raise ValueError(f"Comprimento da palavra deve ser não-negativo (recebeu {word_or_length})")
        length = word_or_length

    gold_set = _normalize_offsets(gold_offsets, length, "gold_offsets")
    pred_set = _normalize_offsets(pred_offsets, length, "pred_offsets")

    if length <= 1:
        # Palavras vazias ou de 1 caractere não possuem cortes internos
        return WordSegmentationResult(
            word_length=length,
            candidate_positions=0,
            tp=0,
            tn=0,
            fp=0,
            fn=0,
            exact_match=(gold_set == pred_set),
        )

    candidate_positions = length - 1
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    tn = candidate_positions - len(gold_set | pred_set)
    exact_match = (gold_set == pred_set)

    return WordSegmentationResult(
        word_length=length,
        candidate_positions=candidate_positions,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        exact_match=exact_match,
    )


def evaluate_corpus_offsets(
    items: Iterable[tuple[str | int, Iterable[int], Iterable[int]]],
) -> OffsetSegmentationConfusion:
    """Avalia e acumula resultados de segmentação sobre um corpus de palavras."""
    if items is None:
        raise TypeError("items não pode ser None")

    words_count = 0
    exact_matches = 0
    candidate_positions = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for idx, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise TypeError(
                f"Cada item do corpus deve ser uma tupla de 3 elementos "
                f"(word_or_length, gold_offsets, pred_offsets), erro no índice {idx}"
            )
        word_or_length, gold, pred = item
        res = evaluate_word_offsets(word_or_length, gold, pred)
        words_count += 1
        if res.exact_match:
            exact_matches += 1
        candidate_positions += res.candidate_positions
        tp += res.tp
        tn += res.tn
        fp += res.fp
        fn += res.fn

    return OffsetSegmentationConfusion(
        words_count=words_count,
        exact_matches=exact_matches,
        candidate_positions=candidate_positions,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def word_exact_match_rate(confusion: OffsetSegmentationConfusion) -> float | None:
    """Proporção de palavras com segmentação exata: exact_matches / words_count.

    Retorna None se não houver palavras no conjunto avaliado.

    Nota semântica sobre o denominador:
        Todas as palavras avaliadas integram o denominador words_count, inclusive palavras
        de comprimento L <= 1 (que não possuem limites internos candidatos e são trivialmente
        exact_match=True na ausência de cortes). Palavras sem candidatos a corte, portanto,
        elevam exact_matches e words_count sem alterar os contadores de offsets
        (candidate_positions=0, TP=TN=FP=FN=0).
    """
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    if confusion.words_count == 0:
        return None
    return confusion.exact_matches / confusion.words_count


def offset_precision(confusion: OffsetSegmentationConfusion) -> float | None:
    """Precisão dos offsets de corte: TP / (TP + FP). Retorna None se nenhum corte foi previsto."""
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    denominator = confusion.predicted_positive
    if denominator == 0:
        return None
    return confusion.tp / denominator


def offset_recall(confusion: OffsetSegmentationConfusion) -> float | None:
    """Recall dos offsets de corte: TP / (TP + FN). Retorna None se não houver cortes no gabarito.

    Nota: Este cálculo corresponde à fórmula matemática que o código legado chamava
    impropriamente de 'boundary_accuracy'.
    """
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    denominator = confusion.support_positive
    if denominator == 0:
        return None
    return confusion.tp / denominator


def offset_specificity(confusion: OffsetSegmentationConfusion) -> float | None:
    """Especificidade dos offsets de corte: TN / (TN + FP).

    Mede a proporção de posições internas de não-corte corretamente identificadas.
    Retorna None se não houver posições negativas (ex.: palavras sem limites internos).
    """
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    denominator = confusion.support_negative
    if denominator == 0:
        return None
    return confusion.tn / denominator


def offset_f1(confusion: OffsetSegmentationConfusion) -> float | None:
    """F1-Score dos offsets de corte: 2TP / (2TP + FP + FN).

    Retorna None se 2TP + FP + FN == 0.
    Retorna 0.0 se TP == 0 e houver cortes no gabarito ou predições.
    """
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    if denominator == 0:
        return None
    return (2.0 * confusion.tp) / denominator


def offset_balanced_accuracy(confusion: OffsetSegmentationConfusion) -> float | None:
    """Balanced Accuracy dos offsets de corte: (offset_recall + offset_specificity) / 2.

    Retorna None se o recall ou a especificidade forem indefinidos.
    """
    rec = offset_recall(confusion)
    spec = offset_specificity(confusion)
    if rec is None or spec is None:
        return None
    return (rec + spec) / 2.0


def offset_segmentation_metrics(confusion: OffsetSegmentationConfusion) -> dict[str, Any]:
    """Resume todas as métricas de segmentação por offsets em um dicionário."""
    if not isinstance(confusion, OffsetSegmentationConfusion):
        raise TypeError("confusion deve ser uma instância de OffsetSegmentationConfusion")
    return {
        "words_count": confusion.words_count,
        "exact_matches": confusion.exact_matches,
        "word_exact_match_rate": word_exact_match_rate(confusion),
        "candidate_positions": confusion.candidate_positions,
        "tp": confusion.tp,
        "tn": confusion.tn,
        "fp": confusion.fp,
        "fn": confusion.fn,
        "support_positive": confusion.support_positive,
        "support_negative": confusion.support_negative,
        "predicted_positive": confusion.predicted_positive,
        "predicted_negative": confusion.predicted_negative,
        "offset_precision": offset_precision(confusion),
        "offset_recall": offset_recall(confusion),
        "offset_specificity": offset_specificity(confusion),
        "offset_f1": offset_f1(confusion),
        "offset_balanced_accuracy": offset_balanced_accuracy(confusion),
    }


def syllable_lengths_to_offsets(syllable_lengths: Sequence[int]) -> list[int]:
    """Converte uma sequência de comprimentos de sílabas em offsets cumulativos de corte.

    Exemplo: [2, 2] -> [2] (para "ca-sa")
    Exemplo: [3, 2, 2, 3] -> [3, 5, 7] (para "com-pu-ta-dor")
    """
    if not isinstance(syllable_lengths, Sequence) or isinstance(syllable_lengths, (str, bytes)):
        raise TypeError("syllable_lengths deve ser uma Sequence de inteiros")
    for idx, length in enumerate(syllable_lengths):
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Comprimento da sílaba no índice {idx} deve ser inteiro positivo (recebeu {length!r})"
            )
    if len(syllable_lengths) <= 1:
        return []
    offsets: list[int] = []
    current = 0
    for length in syllable_lengths[:-1]:
        current += length
        offsets.append(current)
    return offsets
