import json
import math
import tempfile
import unittest
from pathlib import Path

from kinera.language.hybrid_inputs import record_to_hybrid_inputs
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.embedding import Embedding
from kinera.nn.hybrid import HybridRepresentation
from kinera.scripts.build_syllable_vocabulary import build_syllable_vocabulary


class HybridIntegrationTests(unittest.TestCase):
    def test_syllable_vocabulary_builder_writes_artifacts_for_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "records.jsonl"
            vocabulary_path = root / "syllable_vocabulary.json"
            report_path = root / "syllable_vocabulary_report.json"
            records = (
                {"syllables": ["ca", "sa"]},
                {"syllables": ["pa", "to"]},
            )
            source_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            report = build_syllable_vocabulary(
                source_path,
                vocabulary_path,
                report_path,
            )
            vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            persisted_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(vocabulary["version"], 1)
        self.assertEqual(vocabulary["tokens"][0], "<UNK_SYL>")
        self.assertEqual(report["records_read"], 2)
        self.assertEqual(report["invalid_records"], 0)
        self.assertEqual(report["vocabulary_size"], len(vocabulary["tokens"]))
        self.assertEqual(persisted_report, report)

    def test_fixture_annotated_record_runs_through_hybrid_pipeline(self) -> None:
        record = {
            "word_normalized": "a-b",
            "syllables": ["a", "b"],
            "characters": [
                {
                    "char": "a",
                    "syllable_index": 0,
                    "is_vowel": True,
                    "is_alpha": True,
                },
                {"char": "-", "syllable_index": None, "is_separator": True},
                {
                    "char": "b",
                    "syllable_index": 1,
                    "is_consonant": True,
                    "is_alpha": True,
                },
            ],
        }
        char_vocabulary = CharacterVocabulary.from_records([record])
        syllable_vocabulary = SyllableVocabulary.from_records([record])
        representation = HybridRepresentation(
            Embedding(len(char_vocabulary), 4, seed=7),
            Embedding(len(syllable_vocabulary), 4, seed=8),
            feature_dim=10,
        )

        inputs = record_to_hybrid_inputs(
            record,
            char_vocabulary,
            syllable_vocabulary,
        )
        char_ids, syllable_ids, feature_vectors = inputs
        output = representation.forward(*inputs)
        representation.backward(
            [[0.001] * representation.output_dim for _ in output]
        )
        representation.step(0.0001)

        self.assertEqual(len(char_ids), 3)
        self.assertEqual(syllable_ids[1], syllable_vocabulary.unk_id)
        self.assertEqual(len(feature_vectors), 3)
        self.assertTrue(all(len(vector) == 10 for vector in feature_vectors))
        self.assertEqual(len(output), 3)
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
