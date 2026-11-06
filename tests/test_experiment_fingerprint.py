import unittest

from kinera.experiments.comparison import _build_experiment_fingerprint
from kinera.experiments.fingerprint import canonical_sha256
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


class CanonicalFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 1,
                "target_is_syllable_end": 1,
                "source_metadata": "ignored",
            },
            {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 0],
                "context_feature_vectors": [[0.0] * 10, [0.0] * 10],
                "target_char_id": 3,
                "target_syllable_id": 0,
                "target_is_syllable_end": 0,
            },
        ]
        self.validation = [dict(self.train[0], target_char_id=3)]
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(
            ["<UNK_SYL>", "ka"]
        )

    def fingerprint(
        self,
        *,
        train=None,
        validation=None,
        characters=None,
        syllables=None,
        epochs=2,
        learning_rate=0.05,
        max_examples=2,
        variants=("A", "C", "D"),
    ):
        return _build_experiment_fingerprint(
            self.train if train is None else train,
            self.validation if validation is None else validation,
            self.characters if characters is None else characters,
            self.syllables if syllables is None else syllables,
            syllable_vocab_size=2,
            syllable_unknown_id=0,
            epochs=epochs,
            learning_rate=learning_rate,
            max_examples=max_examples,
            variants=variants,
        )

    def test_canonical_hash_is_stable_for_mapping_insertion_order(self) -> None:
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [True, None]}),
            canonical_sha256({"b": [True, None], "a": 1}),
        )

    def test_canonical_hash_preserves_types_and_list_order(self) -> None:
        integer = canonical_sha256(1)
        self.assertNotEqual(integer, canonical_sha256(True))
        self.assertNotEqual(integer, canonical_sha256("1"))
        self.assertNotEqual(canonical_sha256(None), canonical_sha256("None"))
        self.assertNotEqual(canonical_sha256([1, 2]), canonical_sha256([2, 1]))

    def test_canonical_hash_rejects_nonfinite_and_unsupported_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN ou infinito"):
                    canonical_sha256(value)
        with self.assertRaisesRegex(TypeError, "chaves.*string"):
            canonical_sha256({1: "invalid"})
        with self.assertRaisesRegex(TypeError, "tipo não suportado"):
            canonical_sha256((1, 2))

    def test_same_effective_content_ignores_nonconsumed_metadata(self) -> None:
        altered = [dict(self.train[0], source_metadata={"path": "/different"}), self.train[1]]
        self.assertEqual(
            self.fingerprint()["digest"],
            self.fingerprint(train=altered)["digest"],
        )

    def test_target_context_feature_and_order_changes_change_identity(self) -> None:
        cases = {
            "target": [dict(self.train[0], target_syllable_id=0), self.train[1]],
            "context": [dict(self.train[0], context_ids=[0, 4]), self.train[1]],
            "feature": [
                dict(self.train[0], context_feature_vectors=[[1.0] * 10] * 2),
                self.train[1],
            ],
            "order": list(reversed(self.train)),
        }
        baseline = self.fingerprint()["digest"]
        for label, examples in cases.items():
            with self.subTest(label=label):
                self.assertNotEqual(baseline, self.fingerprint(train=examples)["digest"])

    def test_validation_split_and_vocabulary_mappings_change_identity(self) -> None:
        baseline = self.fingerprint()["digest"]
        changed_validation = [dict(self.validation[0], target_char_id=4)]
        reordered_characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "b", "a"]
        )
        reordered_syllables = SyllableVocabulary(["<UNK_SYL>", "ga"])
        self.assertNotEqual(
            baseline,
            self.fingerprint(validation=changed_validation)["digest"],
        )
        self.assertNotEqual(
            baseline,
            self.fingerprint(characters=reordered_characters)["digest"],
        )
        self.assertNotEqual(
            baseline,
            self.fingerprint(syllables=reordered_syllables)["digest"],
        )

    def test_scientific_configuration_changes_identity_and_seeds_are_not_included(self) -> None:
        baseline = self.fingerprint()
        self.assertNotEqual(
            baseline["digest"],
            self.fingerprint(epochs=3)["digest"],
        )
        self.assertNotEqual(
            baseline["digest"],
            self.fingerprint(learning_rate=0.1)["digest"],
        )
        self.assertEqual(
            baseline["digest"],
            self.fingerprint(variants=("A", "C", "D", "B"))["digest"],
        )
        self.assertEqual(
            baseline["variant_inputs"]["A"],
            self.fingerprint(variants=("A", "C", "D", "B"))["variant_inputs"]["A"],
        )

    def test_max_examples_hashes_the_effective_prefix(self) -> None:
        self.assertEqual(
            self.fingerprint(max_examples=1)["components"]["training_examples_sha256"],
            self.fingerprint(train=[self.train[0]], max_examples=1)["components"]["training_examples_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
