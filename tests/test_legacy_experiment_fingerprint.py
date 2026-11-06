import json
import tempfile
import unittest
from pathlib import Path

from kinera.experiments.fingerprint import (
    build_legacy_experiment_fingerprint,
    build_legacy_scientific_configuration,
    fingerprint_components_digest,
    validate_fingerprint_manifest,
)
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


class LegacyExperimentFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            {
                "word_key": "aba",
                "target_char": "a",
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] * 10],
                "target_char_id": 3,
                "target_syllable_id": 1,
                "target_is_syllable_end": True,
            },
            {
                "word_key": "aba",
                "target_char": "b",
                "context_ids": [3, 3],
                "context_syllable_ids": [1, 1],
                "context_feature_vectors": [[1.0] * 10, [1.0] * 10],
                "target_char_id": 4,
                "target_syllable_id": 1,
                "target_is_syllable_end": False,
            },
        ]
        self.validation = [dict(self.train[0], target_char_id=4)]
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(["<UNK_SYL>", "ab"])
        self.configuration = {
            "epochs": 2,
            "learning_rate": 0.05,
            "max_examples": 1,
            "variants": ["A", "B", "C", "D"],
            "phase_protocol_version": 1,
        }

    def fingerprint(self, **changes):
        values = {
            "phase": "5.3_expanded_analysis",
            "train_examples": self.train[:1],
            "validation_examples": self.validation,
            "vocabulary": self.characters,
            "syllable_vocabulary": self.syllables,
            "scientific_configuration": self.configuration,
        }
        values.update(changes)
        return build_legacy_experiment_fingerprint(**values)

    def test_same_effective_inputs_have_stable_versioned_identity(self) -> None:
        self.assertEqual(self.fingerprint(), self.fingerprint())
        identity = self.fingerprint()
        self.assertEqual(identity["fingerprint_schema_version"], 1)
        self.assertEqual(identity["algorithm"], "sha256")
        self.assertEqual(len(identity["digest"]), 64)

    def test_dataset_content_order_split_and_diagnostic_metadata_are_identity(self) -> None:
        original = self.fingerprint()["digest"]
        mutations = [
            {"train_examples": [dict(self.train[0], target_char_id=4)]},
            {"train_examples": [dict(self.train[0], context_ids=[3, 3])]},
            {"train_examples": [dict(self.train[0], word_key="different")]},
            {"validation_examples": [dict(self.validation[0], target_char_id=3)]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertNotEqual(original, self.fingerprint(**mutation)["digest"])
        self.assertNotEqual(
            self.fingerprint(train_examples=self.train)["digest"],
            self.fingerprint(train_examples=list(reversed(self.train)))["digest"],
        )

    def test_vocabulary_order_and_scientific_configuration_change_identity(self) -> None:
        original = self.fingerprint()["digest"]
        self.assertNotEqual(
            original,
            self.fingerprint(
                vocabulary=CharacterVocabulary(
                    ["<START>", "<END>", "<UNK>", "b", "a"]
                )
            )["digest"],
        )
        self.assertNotEqual(
            original,
            self.fingerprint(
                syllable_vocabulary=SyllableVocabulary(["<UNK_SYL>", "ba"])
            )["digest"],
        )
        self.assertNotEqual(
            original,
            self.fingerprint(
                scientific_configuration={**self.configuration, "epochs": 3}
            )["digest"],
        )

    def test_scientific_configuration_excludes_seeds_and_worker_count(self) -> None:
        base = {
            "phase": "5.3_expanded_analysis",
            "variants": ["A", "B", "C", "D"],
            "seeds": [7],
            "epochs": 2,
            "learning_rate": 0.05,
            "max_examples": 10,
            "top_examples": 2,
        }
        config = build_legacy_scientific_configuration(
            base,
            vocabulary=self.characters,
            syllable_vocabulary=self.syllables,
        )
        expanded = build_legacy_scientific_configuration(
            {**base, "seeds": [7, 42]},
            vocabulary=self.characters,
            syllable_vocabulary=self.syllables,
        )
        self.assertEqual(config, expanded)
        self.assertNotIn("workers", config)

    def test_canonical_fingerprint_rejects_nonfinite_dataset_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN ou infinito"):
            self.fingerprint(
                train_examples=[dict(self.train[0], irrelevant_metadata=float("nan"))]
            )

    def test_prefix_identity_ignores_records_outside_max_examples(self) -> None:
        baseline = self.fingerprint(train_examples=self.train[:1])["digest"]
        outside_prefix_changed = [self.train[0], dict(self.train[1], target_char_id=3)]
        self.assertEqual(
            baseline,
            self.fingerprint(train_examples=outside_prefix_changed[:1])["digest"],
        )

    def test_manifest_validation_rejects_unknown_version_bad_digest_and_boolean_version(self) -> None:
        expected = self.fingerprint()
        unknown = {**expected, "fingerprint_schema_version": 2}
        wrong_digest = {**expected, "digest": "0" * 64}
        boolean_version = {**expected, "fingerprint_schema_version": True}
        for manifest in (unknown, wrong_digest, boolean_version):
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    validate_fingerprint_manifest(
                        manifest,
                        expected,
                        component="test phase",
                    )

    def test_manifest_validation_rejects_corrupt_component_digest(self) -> None:
        expected = self.fingerprint()
        components = dict(expected["components"])
        components["training_examples_sha256"] = "x" * 64
        malformed = {
            **expected,
            "components": components,
            "digest": fingerprint_components_digest(components),
        }
        with self.assertRaisesRegex(ValueError, "componentes.*inválidos"):
            validate_fingerprint_manifest(
                malformed,
                expected,
                component="test phase",
            )

    def test_worker_rejects_content_changed_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            character_path = root / "characters.json"
            syllable_path = root / "syllables.json"
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            character_path.write_text(
                json.dumps(self.characters.to_dict()), encoding="utf-8"
            )
            syllable_path.write_text(
                json.dumps(self.syllables.to_dict()), encoding="utf-8"
            )
            train_path.write_text(
                "\n".join(json.dumps(item) for item in self.train) + "\n",
                encoding="utf-8",
            )
            validation_path.write_text(
                json.dumps(self.validation[0]) + "\n", encoding="utf-8"
            )
            payload = {
                "vocabulary": str(character_path),
                "syllable_vocabulary": str(syllable_path),
                "train": str(train_path),
                "validation": str(validation_path),
                "max_examples": 1,
                "scientific_configuration": self.configuration,
            }
            train, validation = self.train[:1], self.validation
            expected = self.fingerprint(
                train_examples=train,
                validation_examples=validation,
            )
            payload["experiment_identity"] = expected
            train_path.write_text(
                json.dumps(dict(self.train[0], target_char_id=4)) + "\n",
                encoding="utf-8",
            )
            from kinera.experiments.fingerprint import load_legacy_experiment_inputs

            with self.assertRaisesRegex(ValueError, "training_examples_sha256"):
                load_legacy_experiment_inputs(
                    payload,
                    phase="5.3_expanded_analysis",
                    require_expected_identity=True,
                )

    def test_records_after_max_examples_prefix_do_not_affect_identity(self) -> None:
        from kinera.experiments.fingerprint import load_legacy_experiment_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            character_path = root / "characters.json"
            syllable_path = root / "syllables.json"
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            character_path.write_text(
                json.dumps(self.characters.to_dict()), encoding="utf-8"
            )
            syllable_path.write_text(
                json.dumps(self.syllables.to_dict()), encoding="utf-8"
            )
            validation_path.write_text(
                json.dumps(self.validation[0]) + "\n", encoding="utf-8"
            )
            payload = {
                "vocabulary": str(character_path),
                "syllable_vocabulary": str(syllable_path),
                "train": str(train_path),
                "validation": str(validation_path),
                "max_examples": 1,
                "scientific_configuration": self.configuration,
            }
            train_path.write_text(
                "\n".join(json.dumps(item) for item in self.train) + "\n",
                encoding="utf-8",
            )
            first = load_legacy_experiment_inputs(
                payload,
                phase="5.3_expanded_analysis",
            )[4]
            changed_tail = [self.train[0], dict(self.train[1], target_char_id=3)]
            train_path.write_text(
                "\n".join(json.dumps(item) for item in changed_tail) + "\n",
                encoding="utf-8",
            )
            second = load_legacy_experiment_inputs(
                payload,
                phase="5.3_expanded_analysis",
            )[4]
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
