"""Testes determinísticos do M1.1: Matriz de confusão, baselines e segmentação por offsets."""

from __future__ import annotations

import math
import sys
import unittest

from kinera.monitor.metrics import (
    BinaryConfusion,
    MetricAggregateSummary,
    OffsetSegmentationConfusion,
    PairedDeltaSummary,
    WordSegmentationResult,
    aggregate_metric_values,
    balanced_accuracy,
    binary_classification_metrics,
    constant_baseline_from_counts,
    evaluate_binary_baselines,
    evaluate_constant_baseline,
    evaluate_corpus_offsets,
    evaluate_majority_baseline,
    evaluate_word_offsets,
    f1_score,
    majority_class,
    mean_std,
    offset_balanced_accuracy,
    offset_f1,
    offset_precision,
    offset_recall,
    offset_segmentation_metrics,
    offset_specificity,
    precision,
    recall,
    specificity,
    summarize_paired_deltas,
    syllable_lengths_to_offsets,
    word_exact_match_rate,
)


class BinaryConfusionValidationTests(unittest.TestCase):
    """Testes de validação de contadores e integridade de BinaryConfusion."""

    def test_valid_initialization_and_properties(self) -> None:
        confusion = BinaryConfusion(tp=10, tn=20, fp=5, fn=15)
        self.assertEqual(confusion.tp, 10)
        self.assertEqual(confusion.tn, 20)
        self.assertEqual(confusion.fp, 5)
        self.assertEqual(confusion.fn, 15)
        self.assertEqual(confusion.support_positive, 25)
        self.assertEqual(confusion.support_negative, 25)
        self.assertEqual(confusion.predicted_positive, 15)
        self.assertEqual(confusion.predicted_negative, 35)
        self.assertEqual(confusion.total, 50)

    def test_rejection_of_boolean_values(self) -> None:
        """Garante que booleanos (True/False) não sejam aceitos como contadores inteiros."""
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=True, tn=10, fp=5, fn=5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=10, tn=False, fp=5, fn=5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=10, tn=10, fp=True, fn=5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=10, tn=10, fp=5, fn=False)  # type: ignore[arg-type]

    def test_rejection_of_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            BinaryConfusion(tp=-1, tn=10, fp=5, fn=5)
        with self.assertRaises(ValueError):
            BinaryConfusion(tp=10, tn=-2, fp=5, fn=5)
        with self.assertRaises(ValueError):
            BinaryConfusion(tp=10, tn=10, fp=-1, fn=5)
        with self.assertRaises(ValueError):
            BinaryConfusion(tp=10, tn=10, fp=5, fn=-3)

    def test_rejection_of_float_and_non_integers(self) -> None:
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=1.5, tn=10, fp=5, fn=5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BinaryConfusion(tp=10, tn="10", fp=5, fn=5)  # type: ignore[arg-type]

    def test_addition_of_confusion_matrices(self) -> None:
        c1 = BinaryConfusion(tp=1, tn=2, fp=3, fn=4)
        c2 = BinaryConfusion(tp=10, tn=20, fp=30, fn=40)
        c_sum = c1 + c2
        self.assertEqual(c_sum, BinaryConfusion(tp=11, tn=22, fp=33, fn=44))

        with self.assertRaises(TypeError):
            _ = c1 + 42  # type: ignore[operator]

    def test_serialization_round_trip(self) -> None:
        confusion = BinaryConfusion(tp=7, tn=13, fp=2, fn=5)
        payload = confusion.to_dict()
        self.assertEqual(payload, {"tp": 7, "tn": 13, "fp": 2, "fn": 5})
        rebuilt = BinaryConfusion.from_dict(payload)
        self.assertEqual(rebuilt, confusion)

        with self.assertRaises(TypeError):
            BinaryConfusion.from_dict("not_a_dict")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BinaryConfusion.from_dict({"tp": 1, "tn": 2})

    def test_from_sequences(self) -> None:
        targets = [1, 1, 0, 0, 1]
        predictions = [1, 0, 0, 1, 1]
        confusion = BinaryConfusion.from_sequences(targets, predictions)
        self.assertEqual(confusion.tp, 2)
        self.assertEqual(confusion.tn, 1)
        self.assertEqual(confusion.fp, 1)
        self.assertEqual(confusion.fn, 1)

    def test_from_sequences_validation(self) -> None:
        with self.assertRaises(ValueError):
            BinaryConfusion.from_sequences([1, 0], [1])
        with self.assertRaises(TypeError):
            BinaryConfusion.from_sequences([True, 0], [1, 0])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            BinaryConfusion.from_sequences([1, 2], [1, 0])
        with self.assertRaises(TypeError):
            BinaryConfusion.from_sequences([1, 0], [1, -1])


class BinaryMetricsFormulasTests(unittest.TestCase):
    """Testes exaustivos das fórmulas de classificação binária e casos-limite."""

    def test_normal_metrics_balanced(self) -> None:
        confusion = BinaryConfusion(tp=20, tn=30, fp=10, fn=5)
        # support_positive = 25, support_negative = 40
        # predicted_positive = 30, predicted_negative = 35
        self.assertAlmostEqual(precision(confusion), 20 / 30)  # ~0.6667
        self.assertAlmostEqual(recall(confusion), 20 / 25)  # 0.8
        self.assertAlmostEqual(specificity(confusion), 30 / 40)  # 0.75
        self.assertAlmostEqual(f1_score(confusion), 40 / 55)  # 2*20 / (40 + 10 + 5) = 40/55
        self.assertAlmostEqual(balanced_accuracy(confusion), (0.8 + 0.75) / 2.0)  # 0.775

        res = binary_classification_metrics(confusion)
        self.assertAlmostEqual(res["precision"], 20 / 30)
        self.assertAlmostEqual(res["recall"], 0.8)
        self.assertAlmostEqual(res["specificity"], 0.75)
        self.assertAlmostEqual(res["f1"], 40 / 55)
        self.assertAlmostEqual(res["balanced_accuracy"], 0.775)

    def test_zero_positive_support(self) -> None:
        """Suporte positivo igual a zero: Recall e Balanced Accuracy devem ser None."""
        confusion = BinaryConfusion(tp=0, tn=40, fp=10, fn=0)
        self.assertEqual(confusion.support_positive, 0)
        self.assertEqual(confusion.support_negative, 50)

        self.assertIsNone(recall(confusion))
        self.assertIsNone(balanced_accuracy(confusion))
        self.assertEqual(precision(confusion), 0.0)  # 0 / 10
        self.assertEqual(specificity(confusion), 40 / 50)  # 0.8
        self.assertEqual(f1_score(confusion), 0.0)  # 2*0 / (0 + 10 + 0) = 0.0

    def test_zero_negative_support(self) -> None:
        """Suporte negativo igual a zero: Especificidade e Balanced Accuracy devem ser None."""
        confusion = BinaryConfusion(tp=30, tn=0, fp=0, fn=10)
        self.assertEqual(confusion.support_negative, 0)
        self.assertEqual(confusion.support_positive, 40)

        self.assertIsNone(specificity(confusion))
        self.assertIsNone(balanced_accuracy(confusion))
        self.assertEqual(recall(confusion), 30 / 40)  # 0.75
        self.assertEqual(precision(confusion), 1.0)  # 30 / 30
        self.assertAlmostEqual(f1_score(confusion), 60 / 70)

    def test_no_predicted_positives(self) -> None:
        """Modelo prediz zero para tudo: Precisão é None; F1 é 0.0 se houver FN."""
        confusion = BinaryConfusion(tp=0, tn=50, fp=0, fn=20)
        self.assertEqual(confusion.predicted_positive, 0)

        self.assertIsNone(precision(confusion))
        self.assertEqual(recall(confusion), 0.0)
        self.assertEqual(specificity(confusion), 1.0)
        self.assertEqual(balanced_accuracy(confusion), 0.5)
        # F1 calculado direto dos contadores: 2*0 / (0 + 0 + 20) = 0.0
        self.assertEqual(f1_score(confusion), 0.0)

    def test_f1_score_indisponivel(self) -> None:
        """F1 é None estritamente quando 2*TP + FP + FN == 0 (ex.: sem positivos reais nem previstos)."""
        confusion = BinaryConfusion(tp=0, tn=50, fp=0, fn=0)
        self.assertIsNone(f1_score(confusion))
        self.assertIsNone(precision(confusion))
        self.assertIsNone(recall(confusion))
        self.assertEqual(specificity(confusion), 1.0)
        self.assertIsNone(balanced_accuracy(confusion))

    def test_all_zeros_confusion(self) -> None:
        confusion = BinaryConfusion(tp=0, tn=0, fp=0, fn=0)
        self.assertIsNone(precision(confusion))
        self.assertIsNone(recall(confusion))
        self.assertIsNone(specificity(confusion))
        self.assertIsNone(f1_score(confusion))
        self.assertIsNone(balanced_accuracy(confusion))

    def test_no_nan_or_infinity(self) -> None:
        """Garante que nenhuma métrica retorne float('nan') ou float('inf')."""
        test_matrices = [
            BinaryConfusion(tp=0, tn=0, fp=0, fn=0),
            BinaryConfusion(tp=0, tn=10, fp=0, fn=0),
            BinaryConfusion(tp=10, tn=0, fp=0, fn=0),
            BinaryConfusion(tp=0, tn=0, fp=10, fn=0),
            BinaryConfusion(tp=0, tn=0, fp=0, fn=10),
        ]
        for conf in test_matrices:
            res = binary_classification_metrics(conf)
            for k, val in res.items():
                if isinstance(val, float):
                    self.assertTrue(
                        math.isfinite(val),
                        f"Campo {k} contém valor não-finito {val!r}",
                    )


class BinaryBaselinesTests(unittest.TestCase):
    """Testes dos baselines determinísticos e verificação da fixture da Fase 5.1."""

    def test_phase51_fixture_547pos_453neg(self) -> None:
        """Fixture histórica da Fase 5.1 (A/seed 7): 547 positivos e 453 negativos.

        Um classificador trivial constante 1 (que prevê sempre positivo) atinge:
        - accuracy = 0.547
        - precision = 0.547 (547 / 1000)
        - recall = 1.0 (547 / 547)
        - specificity = 0.0 (0 / 453)
        - balanced_accuracy = 0.5
        - F1 = 1094 / 1547 ≈ 0.7071751777634131

        Este teste formaliza que esse resultado contextualiza um baseline trivial,
        e NÃO comprova desempenho preditivo real de um modelo neural.
        """
        targets = [1] * 547 + [0] * 453
        baseline_1 = evaluate_constant_baseline(targets, constant_prediction=1)

        self.assertEqual(baseline_1["support_positive"], 547)
        self.assertEqual(baseline_1["support_negative"], 453)
        self.assertEqual(baseline_1["total"], 1000)
        self.assertEqual(baseline_1["confusion"]["tp"], 547)
        self.assertEqual(baseline_1["confusion"]["fp"], 453)
        self.assertEqual(baseline_1["confusion"]["fn"], 0)
        self.assertEqual(baseline_1["confusion"]["tn"], 0)

        self.assertAlmostEqual(baseline_1["precision"], 547 / 1000, places=6)
        self.assertEqual(baseline_1["recall"], 1.0)
        self.assertEqual(baseline_1["specificity"], 0.0)
        self.assertEqual(baseline_1["balanced_accuracy"], 0.5)

        # F1 exato: 2 * 547 / (2 * 547 + 453 + 0) = 1094 / 1547 ≈ 0.707175
        expected_f1 = 1094 / 1547
        self.assertAlmostEqual(baseline_1["f1"], expected_f1, places=6)
        self.assertAlmostEqual(baseline_1["f1"], 0.707175, places=5)

    def test_constant_0_baseline(self) -> None:
        targets = [1] * 30 + [0] * 70
        baseline_0 = evaluate_constant_baseline(targets, constant_prediction=0)

        self.assertEqual(baseline_0["confusion"]["tp"], 0)
        self.assertEqual(baseline_0["confusion"]["tn"], 70)
        self.assertEqual(baseline_0["confusion"]["fp"], 0)
        self.assertEqual(baseline_0["confusion"]["fn"], 30)

        self.assertIsNone(baseline_0["precision"])
        self.assertEqual(baseline_0["recall"], 0.0)
        self.assertEqual(baseline_0["specificity"], 1.0)
        self.assertEqual(baseline_0["balanced_accuracy"], 0.5)
        self.assertEqual(baseline_0["f1"], 0.0)

    def test_majority_class_unbalanced_and_deterministic_tie_break(self) -> None:
        # Desbalanceado classe 1
        self.assertEqual(majority_class([0, 1, 1, 1, 0]), 1)
        self.assertEqual(majority_class({"0": 100, "1": 200}), 1)

        # Desbalanceado classe 0
        self.assertEqual(majority_class([0, 1, 0, 0, 1]), 0)
        self.assertEqual(majority_class({"0": 200, "1": 100}), 0)

        # Empate exato: desempate DETERMINÍSTICO para 0
        self.assertEqual(majority_class([0, 1, 0, 1]), 0)
        self.assertEqual(majority_class({"0": 50, "1": 50}), 0)
        self.assertEqual(majority_class({0: 50, 1: 50}), 0)

    def test_majority_class_validation(self) -> None:
        with self.assertRaises(ValueError):
            majority_class([])
        with self.assertRaises(ValueError):
            majority_class({"0": 0, "1": 0})
        with self.assertRaises(TypeError):
            majority_class("invalid")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            majority_class([True, False])  # type: ignore[list-item]

    def test_evaluate_binary_baselines_bundle(self) -> None:
        train_data = {"0": 60, "1": 40}  # Majoritária = 0
        eval_targets = [1] * 20 + [0] * 80

        res = evaluate_binary_baselines(eval_targets, train_data=train_data)
        self.assertIn("constant_0", res)
        self.assertIn("constant_1", res)
        self.assertIn("majority", res)
        self.assertEqual(res["majority"]["majority_class"], 0)
        self.assertEqual(
            res["majority"]["confusion"],
            res["constant_0"]["confusion"],
        )


class OffsetSegmentationMetricsTests(unittest.TestCase):
    """Testes de avaliação de segmentação silábica por offsets de corte."""

    def test_empty_and_single_character_words(self) -> None:
        # Palavra vazia (L=0)
        res_empty = evaluate_word_offsets("", gold_offsets=[], pred_offsets=[])
        self.assertEqual(res_empty.word_length, 0)
        self.assertEqual(res_empty.candidate_positions, 0)
        self.assertEqual(res_empty.tp, 0)
        self.assertEqual(res_empty.tn, 0)
        self.assertEqual(res_empty.fp, 0)
        self.assertEqual(res_empty.fn, 0)
        self.assertTrue(res_empty.exact_match)

        # Palavra de 1 caractere (L=1)
        res_single = evaluate_word_offsets("a", gold_offsets=[], pred_offsets=[])
        self.assertEqual(res_single.word_length, 1)
        self.assertEqual(res_single.candidate_positions, 0)
        self.assertTrue(res_single.exact_match)

        # Offsets não vazios para palavra L<=1 devem lançar erro
        with self.assertRaises(ValueError):
            evaluate_word_offsets("a", gold_offsets=[1], pred_offsets=[])
        with self.assertRaises(ValueError):
            evaluate_word_offsets("", gold_offsets=[], pred_offsets=[1])

    def test_duplicate_offsets_are_deduplicated(self) -> None:
        """Duplicatas em gold ou pred são normalizadas para conjuntos determinísticos."""
        # Palavra "casa" (L=4): limites em [1, 3]
        res = evaluate_word_offsets(4, gold_offsets=[2, 2], pred_offsets=[2])
        self.assertEqual(res.tp, 1)
        self.assertEqual(res.tn, 2)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 0)
        self.assertTrue(res.exact_match)

    def test_offsets_out_of_bounds_and_invalid_types(self) -> None:
        # Offsets fora de [1, L-1] para L=4 ("casa")
        with self.assertRaises(ValueError):
            evaluate_word_offsets("casa", gold_offsets=[0], pred_offsets=[2])
        with self.assertRaises(ValueError):
            evaluate_word_offsets("casa", gold_offsets=[2], pred_offsets=[4])
        with self.assertRaises(ValueError):
            evaluate_word_offsets("casa", gold_offsets=[-1], pred_offsets=[2])

        # Tipos inválidos
        with self.assertRaises(TypeError):
            evaluate_word_offsets("casa", gold_offsets=[True], pred_offsets=[2])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            evaluate_word_offsets("casa", gold_offsets=[2.5], pred_offsets=[2])  # type: ignore[list-item]

    def test_offset_recall_vs_word_exact_match_divergence(self) -> None:
        """Demonstração da diferença fundamental entre recall de offsets e correspondência exata.

        Palavra: "computador" (L=10)
        Gold: ["com", "pu", "ta", "dor"] -> offsets {3, 5, 7}
        Pred: {3, 5, 7, 8} (contém todos os corretos, mas com um corte espúrio no offset 8)

        - Recall de offsets: 3 / 3 = 1.0 (100% dos cortes de referência encontrados!).
        - Correspondência exata: FALSE (a palavra foi segmentada incorretamente).
        - Precisão de offsets: 3 / 4 = 0.75.
        - F1 de offsets: 6 / (6 + 1 + 0) = 6/7 ≈ 0.8571.
        """
        res = evaluate_word_offsets(
            "computador",
            gold_offsets=[3, 5, 7],
            pred_offsets=[3, 5, 7, 8],
        )
        self.assertEqual(res.word_length, 10)
        self.assertEqual(res.candidate_positions, 9)
        self.assertEqual(res.tp, 3)
        self.assertEqual(res.fp, 1)  # offset 8 é espúrio
        self.assertEqual(res.fn, 0)
        self.assertEqual(res.tn, 5)  # 9 - 4 = 5
        self.assertFalse(res.exact_match)

        # Na agregação de corpus com essa palavra:
        conf = evaluate_corpus_offsets([("computador", [3, 5, 7], [3, 5, 7, 8])])
        self.assertEqual(offset_recall(conf), 1.0)
        self.assertEqual(offset_precision(conf), 0.75)
        self.assertAlmostEqual(offset_f1(conf), 6 / 7)
        self.assertEqual(word_exact_match_rate(conf), 0.0)

    def test_corpus_evaluation_and_metrics_summary(self) -> None:
        corpus = [
            # Palavra 1: segmentação perfeita ("ca-sa", L=4, gold={2}, pred={2})
            ("casa", [2], [2]),
            # Palavra 2: corte omitido ("ba-na-na", L=6, gold={2, 4}, pred={2})
            ("banana", [2, 4], [2]),
            # Palavra 3: corte espúrio ("sol", L=3, gold={}, pred={1})
            ("sol", [], [1]),
        ]
        conf = evaluate_corpus_offsets(corpus)
        # Palavra 1 (L=4, cand=3): TP=1, TN=2, FP=0, FN=0, exact=True
        # Palavra 2 (L=6, cand=5): TP=1, TN=3, FP=0, FN=1, exact=False
        # Palavra 3 (L=3, cand=2): TP=0, TN=1, FP=1, FN=0, exact=False
        # Totais: cand = 10; TP = 2; TN = 6; FP = 1; FN = 1; exact = 1/3
        self.assertEqual(conf.words_count, 3)
        self.assertEqual(conf.exact_matches, 1)
        self.assertEqual(conf.candidate_positions, 10)
        self.assertEqual(conf.tp, 2)
        self.assertEqual(conf.tn, 6)
        self.assertEqual(conf.fp, 1)
        self.assertEqual(conf.fn, 1)

        metrics = offset_segmentation_metrics(conf)
        self.assertAlmostEqual(metrics["word_exact_match_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["offset_precision"], 2 / 3)  # 2 / (2 + 1)
        self.assertAlmostEqual(metrics["offset_recall"], 2 / 3)  # 2 / (2 + 1)
        self.assertAlmostEqual(metrics["offset_specificity"], 6 / 7)  # 6 / (6 + 1)
        self.assertAlmostEqual(metrics["offset_f1"], 4 / 6)  # 2*2 / (4 + 1 + 1) = 4/6
        self.assertAlmostEqual(
            metrics["offset_balanced_accuracy"], (2 / 3 + 6 / 7) / 2.0
        )

    def test_empty_corpus(self) -> None:
        conf = evaluate_corpus_offsets([])
        self.assertEqual(conf.words_count, 0)
        self.assertEqual(conf.candidate_positions, 0)
        self.assertIsNone(word_exact_match_rate(conf))
        self.assertIsNone(offset_precision(conf))
        self.assertIsNone(offset_recall(conf))
        self.assertIsNone(offset_specificity(conf))
        self.assertIsNone(offset_f1(conf))
        self.assertIsNone(offset_balanced_accuracy(conf))

    def test_words_without_candidates_impact_on_exact_match_denominator(self) -> None:
        """Demonstra deterministamente o efeito de palavras sem candidatos (L <= 1) sobre word_exact_match_rate.

        Cenário:
        1. Subconjunto com posições candidatas (L > 1):
           - Palavra 1: "casa" (L=4, cand=3, gold={2}, pred={2}) -> exact_match = True
           - Palavra 2: "banana" (L=6, cand=5, gold={2, 4}, pred={2}) -> exact_match = False
           Taxa com palavras candidatas: 1 acerto / 2 palavras = 0.50 (50%).
           Total de posições candidatas: 3 + 5 = 8.

        2. Adição de palavras sem posições candidatas (L <= 1):
           - Palavra 3: "a" (L=1, cand=0, gold={}, pred={}) -> exact_match = True
           - Palavra 4: "" (L=0, cand=0, gold={}, pred={}) -> exact_match = True

        3. Efeito na agregação do corpus total:
           - Numerador (exact_matches): sobe de 1 para 3 (+2).
           - Denominador (words_count): sobe de 2 para 4 (+2).
           - word_exact_match_rate: sobe de 0.50 para 3/4 = 0.75 (75%).
           - candidate_positions: permanece 8 (inalterado).
           - Contadores de offsets (TP, TN, FP, FN): estritamente inalterados.
           - offset_precision, offset_recall e offset_f1: estritamente inalterados.
        """
        # 1. Corpus apenas com palavras candidatas
        candidate_words = [
            ("casa", [2], [2]),
            ("banana", [2, 4], [2]),
        ]
        conf_candidates = evaluate_corpus_offsets(candidate_words)
        self.assertEqual(conf_candidates.words_count, 2)
        self.assertEqual(conf_candidates.exact_matches, 1)
        self.assertEqual(conf_candidates.candidate_positions, 8)
        self.assertEqual(word_exact_match_rate(conf_candidates), 0.5)

        # 2. Corpus combinado com palavras L <= 1
        full_corpus = [
            ("casa", [2], [2]),
            ("banana", [2, 4], [2]),
            ("a", [], []),
            ("", [], []),
        ]
        conf_full = evaluate_corpus_offsets(full_corpus)
        self.assertEqual(conf_full.words_count, 4)
        self.assertEqual(conf_full.exact_matches, 3)
        self.assertEqual(conf_full.candidate_positions, 8)  # Inalterado
        self.assertEqual(word_exact_match_rate(conf_full), 0.75)  # 3 / 4

        # Comprova invariância exata dos contadores e métricas de offsets:
        self.assertEqual(conf_full.tp, conf_candidates.tp)
        self.assertEqual(conf_full.tn, conf_candidates.tn)
        self.assertEqual(conf_full.fp, conf_candidates.fp)
        self.assertEqual(conf_full.fn, conf_candidates.fn)
        self.assertEqual(offset_precision(conf_full), offset_precision(conf_candidates))
        self.assertEqual(offset_recall(conf_full), offset_recall(conf_candidates))
        self.assertEqual(offset_f1(conf_full), offset_f1(conf_candidates))

    def test_syllable_lengths_to_offsets(self) -> None:
        self.assertEqual(syllable_lengths_to_offsets([2, 2]), [2])
        self.assertEqual(syllable_lengths_to_offsets([3, 2, 2, 3]), [3, 5, 7])
        self.assertEqual(syllable_lengths_to_offsets([5]), [])
        self.assertEqual(syllable_lengths_to_offsets([]), [])

        with self.assertRaises(ValueError):
            syllable_lengths_to_offsets([2, 0])
        with self.assertRaises(ValueError):
            syllable_lengths_to_offsets([2, -1])
        with self.assertRaises(TypeError):
            syllable_lengths_to_offsets("invalid")  # type: ignore[arg-type]


class PureMathAndCpuOnlyTests(unittest.TestCase):
    """Garante pureza e que nenhum módulo CUDA é importado."""

    def test_cpu_only_guarantee(self) -> None:
        import subprocess

        cmd = [
            sys.executable,
            "-c",
            "import sys; import kinera.monitor.metrics; "
            "cuda_mods = [m for m in sys.modules if 'cuda' in m.lower() and m.startswith('kinera.')]; "
            "assert not cuda_mods, f'CUDA imported: {cuda_mods}'",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Falha na garantia CPU-only: {res.stderr}")


class MetricAggregationAndConservationTests(unittest.TestCase):
    """Testes da classificação tripartida e da invariante de conservação em agregação."""

    def test_conservation_invariant_mixed_observations(self) -> None:
        # 10 observações: 5 válidas, 3 ausentes (None), 2 inválidas (bool, NaN)
        obs = [
            (1, 0.8),
            (2, None),
            (3, 0.9),
            (4, float("nan")),
            (5, True),  # bool é inválido
            (6, None),
            (7, 0.7),
            (8, 0.85),
            (9, None),
            (10, 0.75),
        ]
        summary = aggregate_metric_values(obs)
        self.assertEqual(summary.n_values, 5)
        self.assertEqual(summary.n_missing, 3)
        self.assertEqual(summary.n_invalid, 2)
        # Invariante de conservação estrita:
        self.assertEqual(
            summary.n_values + summary.n_missing + summary.n_invalid,
            len(obs),
        )
        self.assertEqual(summary.seeds_used, [1, 3, 7, 8, 10])
        self.assertIsNotNone(summary.mean)
        self.assertIsNotNone(summary.std)
        expected_mean = (0.8 + 0.9 + 0.7 + 0.85 + 0.75) / 5.0
        self.assertAlmostEqual(summary.mean, expected_mean)

    def test_all_valid_observations(self) -> None:
        obs = [(42, 10.0), (43, 20.0), (44, 30.0)]
        summary = aggregate_metric_values(obs)
        self.assertEqual(summary.n_values, 3)
        self.assertEqual(summary.n_missing, 0)
        self.assertEqual(summary.n_invalid, 0)
        self.assertEqual(summary.n_values + summary.n_missing + summary.n_invalid, 3)
        self.assertAlmostEqual(summary.mean, 20.0)
        expected_std = math.sqrt(((10.0 - 20.0) ** 2 + (20.0 - 20.0) ** 2 + (30.0 - 20.0) ** 2) / 3.0)
        self.assertAlmostEqual(summary.std, expected_std)

    def test_all_missing_observations(self) -> None:
        obs = [(1, None), (2, None)]
        summary = aggregate_metric_values(obs)
        self.assertEqual(summary.n_values, 0)
        self.assertEqual(summary.n_missing, 2)
        self.assertEqual(summary.n_invalid, 0)
        self.assertEqual(summary.n_values + summary.n_missing + summary.n_invalid, 2)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.std)
        self.assertEqual(summary.seeds_used, [])

    def test_all_invalid_observations(self) -> None:
        obs = [(1, float("inf")), (2, "texto"), (3, False)]
        summary = aggregate_metric_values(obs)
        self.assertEqual(summary.n_values, 0)
        self.assertEqual(summary.n_missing, 0)
        self.assertEqual(summary.n_invalid, 3)
        self.assertEqual(summary.n_values + summary.n_missing + summary.n_invalid, 3)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.std)

    def test_empty_observations(self) -> None:
        summary = aggregate_metric_values([])
        self.assertEqual(summary.n_values, 0)
        self.assertEqual(summary.n_missing, 0)
        self.assertEqual(summary.n_invalid, 0)
        self.assertEqual(summary.n_values + summary.n_missing + summary.n_invalid, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.std)


class PairedDeltaTests(unittest.TestCase):
    """Testes de deltas pareados com classificação mutuamente exclusiva e invariantes."""

    def test_paired_delta_conservation_invariant(self) -> None:
        # Base: seeds 1..6
        base = {1: 1.0, 2: 1.5, 3: None, 4: 2.0, 5: float("nan"), 6: 3.0}
        # Cand: seeds 1..7 (seed 7 não está em base)
        cand = {1: 0.8, 2: None, 3: 1.0, 4: 1.2, 5: 1.0, 6: True, 7: 0.5}

        # Common seeds: {1, 2, 3, 4, 5, 6} -> total 6
        # Pair 1: base=1.0, cand=0.8 -> pairs (válido)
        # Pair 2: cand=None -> missing_pairs
        # Pair 3: base=None -> missing_pairs
        # Pair 4: base=2.0, cand=1.2 -> pairs (válido)
        # Pair 5: base=nan -> invalid_pairs (nenhum é None)
        # Pair 6: cand=True (bool) -> invalid_pairs (nenhum é None)
        summary = summarize_paired_deltas(base, cand, orientation="base - candidate", unit="loss")
        self.assertEqual(summary.n_pairs, 2)
        self.assertEqual(summary.n_missing_pairs, 2)
        self.assertEqual(summary.n_invalid_pairs, 2)
        # Invariante de conservação:
        self.assertEqual(
            summary.n_pairs + summary.n_missing_pairs + summary.n_invalid_pairs,
            6,
        )
        self.assertEqual(summary.seeds_used, [1, 4])
        self.assertEqual(summary.n_pairs, len(summary.seeds_used))
        # Deltas: seed 1: 1.0 - 0.8 = 0.2; seed 4: 2.0 - 1.2 = 0.8
        expected_mean = (0.2 + 0.8) / 2.0
        self.assertAlmostEqual(summary.mean, expected_mean)

    def test_paired_delta_orientations(self) -> None:
        base = {42: 0.5}
        cand = {42: 0.8}

        # Loss (base - candidate: melhora se cand for menor)
        s_loss = summarize_paired_deltas(base, cand, orientation="base - candidate")
        self.assertAlmostEqual(s_loss.mean, 0.5 - 0.8)  # -0.3 (candidato piorou em loss)

        # Accuracy (candidate - base: melhora se cand for maior)
        s_acc = summarize_paired_deltas(base, cand, orientation="candidate - base")
        self.assertAlmostEqual(s_acc.mean, 0.8 - 0.5)  # +0.3 (candidato melhorou em accuracy)

    def test_no_common_seeds(self) -> None:
        base = {1: 0.5, 2: 0.6}
        cand = {3: 0.7, 4: 0.8}
        summary = summarize_paired_deltas(base, cand)
        self.assertEqual(summary.n_pairs, 0)
        self.assertEqual(summary.n_missing_pairs, 0)
        self.assertEqual(summary.n_invalid_pairs, 0)
        self.assertEqual(summary.seeds_used, [])
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.std)


class EvaluatorM12IntegrationTests(unittest.TestCase):
    """Testes de integração do avaliador neural, baselines sem vazamento e campos legados."""

    def test_evaluator_majority_baseline_without_leakage(self) -> None:
        from kinera.language.vocabulary import CharacterVocabulary
        from kinera.nn.models.hybrid import AutoregressiveModel
        from kinera.training.evaluator import evaluate_multitask_examples

        vocab = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b", "c"])
        model = AutoregressiveModel(len(vocab), 10, seed=7)
        # Exemplo com target_is_syllable_end = 1
        examples = [{
            "context_ids": [0, 3],
            "context_syllable_ids": [0, 0],
            "context_feature_vectors": [[0.0] * 10] * 2,
            "target_char_id": 4,
            "target_char": "b",
            "target_syllable_id": 1,
            "target_syllable_index": 0,
            "target_is_syllable_end": 1,
        }]

        # 1. Sem train_boundary_distribution: majority baseline deve ser None (sem vazamento)
        res_no_leak = evaluate_multitask_examples(model, examples, vocab)
        self.assertIsNone(res_no_leak["boundary_baseline_majority"])
        self.assertIn("boundary_baseline_constant_0", res_no_leak)
        self.assertIn("boundary_baseline_constant_1", res_no_leak)
        self.assertIn("boundary_specificity", res_no_leak)
        self.assertIn("boundary_balanced_accuracy", res_no_leak)
        self.assertIn("boundary_true_negative", res_no_leak)

        # 2. Com train_boundary_distribution onde classe majoritária é 0:
        # No exemplo de eval, target é 1 -> previsão constante 0 acerta 0/1 -> acurácia 0.0
        train_dist_0 = {"0": 100, "1": 10}
        res_with_train = evaluate_multitask_examples(
            model,
            examples,
            vocab,
            train_boundary_distribution=train_dist_0,
        )
        self.assertEqual(res_with_train["boundary_baseline_majority"], 0.0)

        # 3. Com train_boundary_distribution onde classe majoritária é 1:
        # Previsão constante 1 acerta 1/1 -> acurácia 1.0
        train_dist_1 = {"0": 10, "1": 100}
        res_with_train_1 = evaluate_multitask_examples(
            model,
            examples,
            vocab,
            train_boundary_distribution=train_dist_1,
        )
        self.assertEqual(res_with_train_1["boundary_baseline_majority"], 1.0)


if __name__ == "__main__":
    unittest.main()
