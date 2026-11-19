"""R2.3 — fail-closed compatibility and integrity gates for hybrid CPU snapshots."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinera.backends.cpu.backend import CPUBackend
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.nn.neuron import Neuron
from kinera.nn.linear import Linear
from kinera.training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_neuron,
    load_linear,
    save_checkpoint,
)
from kinera.training.hybrid_checkpoint import (
    load_autoregressive_checkpoint,
    save_autoregressive_checkpoint,
)


class HybridCheckpointIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        work = tempfile.TemporaryDirectory(prefix="kinera-r2-3-")
        self.addCleanup(work.cleanup)
        self.path = Path(work.name) / "model.json"
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(["<UNK_SYL>", "ka", "ke"])
        self.model = AutoregressiveModel(
            char_vocab_size=5, syllable_vocab_size=3,
            char_embedding_dim=3, syllable_embedding_dim=4,
            feature_dim=2, seed=17, backend=CPUBackend(),
        )
        save_autoregressive_checkpoint(
            self.path, self.model,
            character_vocabulary=self.characters,
            syllable_vocabulary=self.syllables,
            metadata={"origin": "R2.3"},
        )
        self.valid_payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.original = self.path.read_bytes()

    def _load(self) -> tuple[AutoregressiveModel, dict]:
        return load_autoregressive_checkpoint(
            self.path,
            expected_character_vocabulary=self.characters,
            expected_syllable_vocabulary=self.syllables,
        )

    def _reject(self, payload: object, expected: str) -> None:
        self.path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(CheckpointError, expected):
            self._load()
        self.assertEqual(self.path.read_bytes(), before)

    def test_truncated_missing_invalid_utf8_and_nonobject_json(self) -> None:
        for raw in ('{"format":', '', '[]', '"text"', 'null'):
            with self.subTest(raw=raw):
                self.path.write_text(raw, encoding="utf-8")
                with self.assertRaises(CheckpointError):
                    self._load()
        self.path.write_bytes(b"\xff\xfe\x00")
        with self.assertRaises(CheckpointError):
            self._load()

    def test_rejects_format_version_type_and_resume_state(self) -> None:
        invalid = [
            ("format", "kinera.checkpoint", "formato"),
            ("version", 2, "versão"),
            ("version", True, "versão"),
            ("training", {"epoch": 1}, "R2.4"),
            ("metadata", [], "metadata"),
        ]
        for field, value, error in invalid:
            with self.subTest(field=field, value=value):
                payload = copy.deepcopy(self.valid_payload)
                payload[field] = value
                self._reject(payload, error)
        for field in ("identity", "training", "model"):
            with self.subTest(missing=field):
                payload = copy.deepcopy(self.valid_payload)
                del payload[field]
                self._reject(payload, "checkpoint")
        payload = copy.deepcopy(self.valid_payload)
        payload["unexpected"] = "field"
        self._reject(payload, "checkpoint")
        payload = copy.deepcopy(self.valid_payload)
        payload["model"]["type"] = "linear"
        self._reject(payload, "tipo")

    def test_rejects_missing_extra_nested_fields_and_bad_dimensions(self) -> None:
        paths = [
            (("model", "architecture"), "model"),
            (("model", "parameters"), "model"),
            (("model", "architecture", "feature_dim"), "architecture"),
            (("model", "parameters", "char_embedding", "weights"), "char_embedding"),
            (("model", "parameters", "heads", "char", "bias"), "heads.char"),
            (("model", "parameters", "heads", "boundary"), "heads"),
        ]
        for keys, error in paths:
            with self.subTest(missing=keys):
                payload = copy.deepcopy(self.valid_payload)
                parent = payload
                for key in keys[:-1]:
                    parent = parent[key]
                del parent[keys[-1]]
                self._reject(payload, error)
        for bad in (0, -1, True, "2", 1.5):
            with self.subTest(feature_dim=bad):
                payload = copy.deepcopy(self.valid_payload)
                payload["model"]["architecture"]["feature_dim"] = bad
                self._reject(payload, "feature_dim")
        payload = copy.deepcopy(self.valid_payload)
        payload["model"]["parameters"]["extra"] = []
        self._reject(payload, "parameters")

    def test_rejects_parameter_shapes_types_nonfinite_and_incoherent_architecture(self) -> None:
        changes = [
            (("char_embedding", "weights"), [], "shape"),
            (("syllable_embedding", "weights"), [[0.1]], "shape"),
            (("heads", "char", "bias"), [], "shape"),
            (("heads", "boundary", "weights"), [[], []], "shape"),
        ]
        for keys, value, error in changes:
            with self.subTest(path=keys):
                payload = copy.deepcopy(self.valid_payload)
                current = payload["model"]["parameters"]
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = value
                self._reject(payload, error)
        payload = copy.deepcopy(self.valid_payload)
        payload["model"]["architecture"]["char_embedding_dim"] += 1
        self._reject(payload, "shape")
        for value in (True, "1.5", math.nan, math.inf, -math.inf):
            with self.subTest(value=repr(value)):
                payload = copy.deepcopy(self.valid_payload)
                payload["model"]["parameters"]["heads"]["boundary"]["bias"][0] = value
                self._reject(payload, "numérico|finito|constante")

    def test_vocabulary_identity_missing_corrupt_and_same_length_remap(self) -> None:
        payload = copy.deepcopy(self.valid_payload)
        del payload["identity"]
        self._reject(payload, "checkpoint")
        for field, value in (
            ("algorithm", "md5"),
            ("character_sha256", "0" * 64),
            ("syllable_sha256", "1" * 64),
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.valid_payload)
                payload["identity"][field] = value
                self._reject(payload, "algoritmo|identidade|digest")
        payload = copy.deepcopy(self.valid_payload)
        payload["identity"]["character_vocabulary"]["tokens"][3:5] = ["b", "a"]
        self._reject(payload, "identidade|digest")

        self.path.write_bytes(self.original)
        changed = CharacterVocabulary(["<START>", "<END>", "<UNK>", "b", "a"])
        with self.assertRaisesRegex(CheckpointError, "identidade"):
            load_autoregressive_checkpoint(
                self.path,
                expected_character_vocabulary=changed,
                expected_syllable_vocabulary=self.syllables,
            )

    def test_duplicate_json_keys_must_not_be_silently_overwritten(self) -> None:
        raw = self.original.decode("utf-8").rstrip()
        # Duplicated top-level "format", with identical values, previously
        # passed json.loads due to last-write-wins handling.
        raw = raw[:-1] + ', "format": "kinera.hybrid.checkpoint"}'
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(CheckpointError, "duplicada"):
            self._load()

    def test_replace_failure_keeps_old_bytes_and_cleans_temporary_file(self) -> None:
        self.model.heads.char.bias[0] += 0.2
        with patch(
            "kinera.training.checkpoint.os.replace",
            side_effect=OSError("injected atomic replace failure"),
        ):
            with self.assertRaisesRegex(CheckpointError, "não foi possível salvar"):
                save_autoregressive_checkpoint(
                    self.path, self.model,
                    character_vocabulary=self.characters,
                    syllable_vocabulary=self.syllables,
                )
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(list(self.path.parent.glob(".model.json.*.tmp")), [])
        restored, _ = self._load()
        self.assertIsInstance(restored, AutoregressiveModel)

    def test_metadata_failure_does_not_publish_partial_checkpoint(self) -> None:
        for bad in ({"bad": object()}, {"bad": math.nan}):
            with self.subTest(bad=str(bad)):
                with self.assertRaises(CheckpointError):
                    save_autoregressive_checkpoint(
                        self.path, self.model,
                        character_vocabulary=self.characters,
                        syllable_vocabulary=self.syllables,
                        metadata=bad,
                    )
                self.assertEqual(self.path.read_bytes(), self.original)

    def test_explicit_non_cpu_save_is_rejected_without_device_execution(self) -> None:
        backend = self.model.backend
        try:
            self.model.backend = object()  # Stub only; no CUDA/OpenCL instantiated.
            with self.assertRaisesRegex(CheckpointError, "CPUBackend"):
                save_autoregressive_checkpoint(
                    self.path, self.model,
                    character_vocabulary=self.characters,
                    syllable_vocabulary=self.syllables,
                )
        finally:
            self.model.backend = backend
        self.assertEqual(self.path.read_bytes(), self.original)
        loaded, _ = self._load()
        self.assertIsInstance(loaded.backend, CPUBackend)

    def test_legacy_v1_static_fixtures_remain_readable_without_migration(self) -> None:
        fixtures = (
            ({
                "format": "kinera.checkpoint", "version": 1,
                "model": {
                    "type": "neuron", "input_size": 2,
                    "activation": "linear", "negative_slope": 0.01,
                    "weights": [0.25, -0.5], "bias": 0.125,
                },
            }, Neuron, load_neuron),
            ({
                "format": "kinera.checkpoint", "version": 1,
                "model": {
                    "type": "linear", "input_size": 2, "output_size": 2,
                    "weights": [[1.0, 0.0], [0.0, -1.0]],
                    "bias": [0.5, -0.25],
                },
            }, Linear, load_linear),
        )
        for payload, cls, specific_loader in fixtures:
            with self.subTest(model=cls.__name__):
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                before = self.path.read_bytes()
                generic, generic_info = load_checkpoint(self.path)
                specific, specific_info = specific_loader(self.path)
                self.assertIsInstance(generic, cls)
                self.assertIsInstance(specific, cls)
                self.assertEqual(generic_info, {"training": {}, "metadata": {}})
                self.assertEqual(generic_info, specific_info)
                with self.assertRaisesRegex(CheckpointError, "formato"):
                    self._load()
                self.assertEqual(self.path.read_bytes(), before)

        # The historical generic writer still emits legacy v1.
        save_checkpoint(self.path, Neuron(input_size=1, seed=3))
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["version"], 1
        )


if __name__ == "__main__":
    unittest.main()
