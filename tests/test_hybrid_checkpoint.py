"""R2.2: round-trip of all trainable parameters in the CPU hybrid model."""

import json
import tempfile
import unittest
from pathlib import Path

from kinera.backends.cpu.backend import CPUBackend
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.checkpoint import CheckpointError, load_checkpoint
from kinera.training.hybrid_checkpoint import (
    HYBRID_CHECKPOINT_FORMAT,
    HYBRID_CHECKPOINT_VERSION,
    load_autoregressive_checkpoint,
    save_autoregressive_checkpoint,
)


class HybridCheckpointRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(["<UNK_SYL>", "ka", "ke"])
        self.model = AutoregressiveModel(
            char_vocab_size=len(self.characters),
            syllable_vocab_size=len(self.syllables),
            char_embedding_dim=3,
            syllable_embedding_dim=4,
            feature_dim=2,
            seed=11,
            backend=CPUBackend(),
        )
        self.inputs = ([3, 4], [1, 2], [[1.0, 0.0], [0.0, 1.0]])

    def _parameters(self, model: AutoregressiveModel) -> dict[str, object]:
        return {
            "char_embedding": [
                row.copy() for row in model.representation.char_embedding.weights
            ],
            "syllable_embedding": [
                row.copy() for row in model.representation.syllable_embedding.weights
            ],
            "heads": {
                name: {
                    "weights": [row.copy() for row in getattr(model.heads, name).weights],
                    "bias": getattr(model.heads, name).bias.copy(),
                }
                for name in ("char", "syllable", "boundary")
            },
        }

    def test_training_changes_all_parts_and_round_trip_preserves_three_heads(self) -> None:
        before = self._parameters(self.model)
        report = self.model.multitask_train_step(
            *self.inputs,
            targets={"char": 4, "syllable": 2, "boundary": 1},
            learning_rate=0.08,
        )
        self.assertGreater(report["total_loss"], 0.0)
        after = self._parameters(self.model)
        self.assertNotEqual(before["char_embedding"], after["char_embedding"])
        self.assertNotEqual(before["syllable_embedding"], after["syllable_embedding"])
        for name in ("char", "syllable", "boundary"):
            self.assertNotEqual(
                before["heads"][name]["weights"], after["heads"][name]["weights"]
            )
            self.assertNotEqual(
                before["heads"][name]["bias"], after["heads"][name]["bias"]
            )

        expected_heads = self.model.forward_heads(*self.inputs)
        expected_prediction = self.model.predict(*self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "hybrid.json"
            save_autoregressive_checkpoint(
                path,
                self.model,
                character_vocabulary=self.characters,
                syllable_vocabulary=self.syllables,
                metadata={"purpose": "R2.2", "seed": 11},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], HYBRID_CHECKPOINT_FORMAT)
            self.assertEqual(payload["version"], HYBRID_CHECKPOINT_VERSION)
            self.assertIsNone(payload["training"])
            self.assertNotIn("last_logits", path.read_text(encoding="utf-8"))
            restored, info = load_autoregressive_checkpoint(
                path,
                expected_character_vocabulary=self.characters,
                expected_syllable_vocabulary=self.syllables,
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

        self.assertIsNot(restored, self.model)
        self.assertIsInstance(restored.backend, CPUBackend)
        self.assertEqual(self._parameters(restored), after)
        self.assertEqual(restored.representation.output_dim, 9)
        self.assertIs(restored.output_layer, restored.heads.char)
        self.assertIsNone(restored.last_logits)
        self.assertIsNone(restored.last_pooled)
        self.assertIsNone(restored.last_context_size)
        self.assertIsNone(restored.heads.char.last_input)
        self.assertIsNone(restored.representation.char_embedding.last_ids)
        self.assertEqual(info, {"training": None, "metadata": {"purpose": "R2.2", "seed": 11}})
        with self.assertRaises(RuntimeError):
            restored.heads.char.step(0.1)
        self.assertEqual(restored.forward_heads(*self.inputs), expected_heads)
        self.assertEqual(restored.predict(*self.inputs), expected_prediction)

    def test_equal_vocab_sizes_do_not_allow_id_remapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.json"
            save_autoregressive_checkpoint(
                path,
                self.model,
                character_vocabulary=self.characters,
                syllable_vocabulary=self.syllables,
            )
            original = path.read_bytes()
            changed_characters = CharacterVocabulary(
                ["<START>", "<END>", "<UNK>", "b", "a"]
            )
            changed_syllables = SyllableVocabulary(["<UNK_SYL>", "ke", "ka"])
            for chars, syllables in (
                (changed_characters, self.syllables),
                (self.characters, changed_syllables),
            ):
                with self.subTest(characters=chars.tokens, syllables=syllables.tokens):
                    with self.assertRaisesRegex(CheckpointError, "identidade"):
                        load_autoregressive_checkpoint(
                            path,
                            expected_character_vocabulary=chars,
                            expected_syllable_vocabulary=syllables,
                        )
                    self.assertEqual(path.read_bytes(), original)

    def test_hybrid_schema_cannot_be_silently_read_as_legacy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.json"
            save_autoregressive_checkpoint(
                path,
                self.model,
                character_vocabulary=self.characters,
                syllable_vocabulary=self.syllables,
            )
            with self.assertRaisesRegex(CheckpointError, "formato"):
                load_checkpoint(path)

    def test_invalid_model_parameters_do_not_publish_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.json"
            save_autoregressive_checkpoint(
                path,
                self.model,
                character_vocabulary=self.characters,
                syllable_vocabulary=self.syllables,
            )
            original = path.read_bytes()
            self.model.heads.boundary.bias[0] = float("nan")
            with self.assertRaisesRegex(CheckpointError, "finito"):
                save_autoregressive_checkpoint(
                    path,
                    self.model,
                    character_vocabulary=self.characters,
                    syllable_vocabulary=self.syllables,
                )
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
