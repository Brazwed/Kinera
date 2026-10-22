"""Métricas científicas puras, baselines determinísticos e avaliação de segmentação.

Este subpacote concentra as fórmulas matemáticas de qualidade, confusão, baselines
e agregações estatísticas do Kinera, com garantia de:
- Execução puramente CPU-only sem dependências externas (apenas Python stdlib).
- Independência completa de modelos neurais, backends CUDA, trainers ou telemetria.
- Representação explícita de valores indisponíveis por None (nunca NaN ou Infinity).

Superfície pública:
- M0 (Legado / Agregação / Aritmética segura):
  - ``mean_std(values)``: Média e desvio-padrão populacional.
  - ``safe_ratio(...)``: Razão com fallback configurável para divisão por zero.
  - ``mean_from_total(...)``: Divisão direta total / count.
- M1.1 (Matriz de Confusão e Classificação Binária):
  - ``BinaryConfusion``: Estrutura imutável com contadores TP, TN, FP, FN e propriedades.
  - ``precision(confusion)``: Precisão binária.
  - ``recall(confusion)``: Recall binário.
  - ``specificity(confusion)``: Especificidade binária.
  - ``f1_score(confusion)``: F1-score binário direto dos contadores.
  - ``balanced_accuracy(confusion)``: Acurácia balanceada (média de recall e especificidade).
  - ``binary_classification_metrics(confusion)``: Resumo de todas as métricas binárias.
- M1.1 (Baselines Determinísticos de Referência):
  - ``evaluate_constant_baseline(...)``: Avaliação de previsão constante 0 ou 1.
  - ``constant_baseline_from_counts(...)``: Baseline constante a partir de contagens de suporte.
  - ``majority_class(...)``: Identificação de classe majoritária com desempate determinístico para 0.
  - ``evaluate_majority_baseline(...)``: Avaliação do baseline majoritário do treino.
  - ``evaluate_binary_baselines(...)``: Conjunto completo de baselines determinísticos.
- M1.1 (Segmentação Silábica por Offsets):
  - ``WordSegmentationResult``: Resultado por palavra individual.
  - ``OffsetSegmentationConfusion``: Matriz de confusão de offsets em nível de corpus.
  - ``evaluate_word_offsets(...)``: Avaliação de offsets para uma palavra.
  - ``evaluate_corpus_offsets(...)``: Avaliação acumulada sobre um corpus de palavras.
  - ``syllable_lengths_to_offsets(...)``: Conversão de comprimentos silábicos em limites internos.
  - ``word_exact_match_rate(...)``: Taxa de palavras com segmentação exata.
  - ``offset_precision(...)``: Precisão dos offsets de corte.
  - ``offset_recall(...)``: Recall dos offsets de corte.
  - ``offset_specificity(...)``: Especificidade dos offsets de corte.
  - ``offset_f1(...)``: F1 dos offsets de corte.
  - ``offset_balanced_accuracy(...)``: Acurácia balanceada dos offsets de corte.
  - ``offset_segmentation_metrics(...)``: Resumo completo de métricas de segmentação.
"""

from kinera.monitor.metrics.aggregation import (
    MetricAggregateSummary,
    PairedDeltaSummary,
    aggregate_metric_values,
    mean_std,
    summarize_paired_deltas,
)
from kinera.monitor.metrics.baselines import (
    constant_baseline_from_counts,
    evaluate_binary_baselines,
    evaluate_constant_baseline,
    evaluate_majority_baseline,
    majority_class,
)
from kinera.monitor.metrics.confusion import (
    BinaryConfusion,
    balanced_accuracy,
    binary_classification_metrics,
    f1_score,
    precision,
    recall,
    specificity,
)
from kinera.monitor.metrics.core import mean_from_total, safe_ratio
from kinera.monitor.metrics.segmentation import (
    OffsetSegmentationConfusion,
    WordSegmentationResult,
    evaluate_corpus_offsets,
    evaluate_word_offsets,
    offset_balanced_accuracy,
    offset_f1,
    offset_precision,
    offset_recall,
    offset_segmentation_metrics,
    offset_specificity,
    syllable_lengths_to_offsets,
    word_exact_match_rate,
)

__all__ = [
    # M0
    "mean_std",
    "mean_from_total",
    "safe_ratio",
    # M1.2 Agregação e deltas pareados
    "MetricAggregateSummary",
    "aggregate_metric_values",
    "PairedDeltaSummary",
    "summarize_paired_deltas",
    # M1.1 Confusão e classificação
    "BinaryConfusion",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "balanced_accuracy",
    "binary_classification_metrics",
    # M1.1 Baselines
    "constant_baseline_from_counts",
    "evaluate_constant_baseline",
    "majority_class",
    "evaluate_majority_baseline",
    "evaluate_binary_baselines",
    # M1.1 Segmentação por offsets
    "WordSegmentationResult",
    "OffsetSegmentationConfusion",
    "evaluate_word_offsets",
    "evaluate_corpus_offsets",
    "syllable_lengths_to_offsets",
    "word_exact_match_rate",
    "offset_precision",
    "offset_recall",
    "offset_specificity",
    "offset_f1",
    "offset_balanced_accuracy",
    "offset_segmentation_metrics",
]
