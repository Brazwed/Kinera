import json
import math
import unittest
from pathlib import Path

from language.hybrid_inputs import record_to_hybrid_inputs
from language.vocabulary import CharacterVocabulary, SyllableVocabulary
from nn.embedding import Embedding
from nn.hybrid import HybridRepresentation


class HybridIntegrationTests(unittest.TestCase):
    def test_real_vocabulary_artifact_has_expected_metadata(self) -> None:
        root = Path(__file__).parents[1]
        vocabulary = json.loads(
            (root / "data/processed/syllable_vocabulary.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (root / "data/processed/syllable_vocabulary_report.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(vocabulary["version"], 1)
        self.assertEqual(vocabulary["tokens"][0], "<UNK_SYL>")
        self.assertEqual(report["records_read"], 181560)
        self.assertEqual(report["invalid_records"], 0)
        self.assertEqual(report["vocabulary_size"], len(vocabulary["tokens"]))

    def test_real_annotated_records_run_through_hybrid_pipeline(self) -> None:
        root = Path(__file__).parents[1]
        char_vocabulary = CharacterVocabulary.from_dict(
            json.loads(
                (root / "data/processed/vocabulary.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        syllable_vocabulary = SyllableVocabulary.from_dict(
            json.loads(
                (root / "data/processed/syllable_vocabulary.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        representation = HybridRepresentation(
            Embedding(len(char_vocabulary), 4, seed=7),
            Embedding(len(syllable_vocabulary), 4, seed=8),
            feature_dim=10,
        )
        processed_positions = 0
        with (root / "data/processed/annotated_syllable_records.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle):
                if line_number >= 64:
                    break
                record = json.loads(line)
                inputs = record_to_hybrid_inputs(
                    record, char_vocabulary, syllable_vocabulary
                )
                output = representation.forward(*inputs)
                representation.backward(
                    [[0.001] * representation.output_dim for _ in output]
                )
                representation.step(0.0001)
                processed_positions += len(output)

        self.assertGreater(processed_positions, 64)
        self.assertEqual(representation.output_dim, 18)


class HybridGradientTests(unittest.TestCase):
    def test_embedding_gradients_match_finite_difference(self) -> None:
        char_embedding = Embedding(vocab_size=2, embedding_dim=2, seed=1)
        syllable_embedding = Embedding(vocab_size=2, embedding_dim=2, seed=2)
        char_embedding.weights = [[0.1, -0.2], [0.3, 0.4]]
        syllable_embedding.weights = [[-0.5, 0.6], [0.7, -0.8]]
        representation = HybridRepresentation(
            char_embedding,
            syllable_embedding,
            feature_dim=1,
        )
        char_ids = [1]
        syllable_ids = [0]
        features = [[0.25]]
        coefficients = [[1.2, -0.7, 0.4, 2.1, -1.5]]

        representation.forward(char_ids, syllable_ids, features)
        representation.backward(coefficients)
        analytical_char = char_embedding.grad_weights[1][0]
        analytical_syllable = syllable_embedding.grad_weights[0][1]

        def loss() -> float:
            output = representation.forward(char_ids, syllable_ids, features)[0]
            return sum(value * coefficient for value, coefficient in zip(output, coefficients[0]))

        epsilon = 1e-6
        original_char = char_embedding.weights[1][0]
        char_embedding.weights[1][0] = original_char + epsilon
        plus_char = loss()
        char_embedding.weights[1][0] = original_char - epsilon
        minus_char = loss()
        char_embedding.weights[1][0] = original_char

        original_syllable = syllable_embedding.weights[0][1]
        syllable_embedding.weights[0][1] = original_syllable + epsilon
        plus_syllable = loss()
        syllable_embedding.weights[0][1] = original_syllable - epsilon
        minus_syllable = loss()
        syllable_embedding.weights[0][1] = original_syllable

        numerical_char = (plus_char - minus_char) / (2 * epsilon)
        numerical_syllable = (plus_syllable - minus_syllable) / (2 * epsilon)
        self.assertTrue(math.isclose(analytical_char, numerical_char, rel_tol=1e-6))
        self.assertTrue(math.isclose(analytical_syllable, numerical_syllable, rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
