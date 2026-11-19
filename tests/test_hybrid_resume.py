"""R2.4: epoch-boundary deterministic CPU resume, without pipeline expansion."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinera.backends.cpu.backend import CPUBackend
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.checkpoint import CheckpointError
from kinera.training.hybrid_checkpoint import (
    load_autoregressive_checkpoint,
    save_autoregressive_checkpoint,
)
from kinera.training.hybrid_resume import (
    load_hybrid_training_checkpoint,
    save_hybrid_training_checkpoint,
    train_hybrid_epochs,
)
from kinera.training.trainer import train_examples, train_multitask_examples


class HybridResumeEpochTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(["<UNK_SYL>", "ka", "ke"])
        self.examples = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 1,
                "target_is_syllable_end": 1,
            },
            {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 2],
                "context_feature_vectors": [[0.0, 1.0] + [0.0] * 8, [0.0] * 10],
                "target_char_id": 1,
                "target_syllable_id": 2,
                "target_is_syllable_end": 0,
            },
            {
                "context_ids": [0, 4],
                "context_syllable_ids": [0, 2],
                "context_feature_vectors": [[0.0] * 10, [1.0, 1.0] + [0.0] * 8],
                "target_char_id": 3,
                "target_syllable_id": 0,
                "target_is_syllable_end": 1,
            },
        ]
        directory = tempfile.TemporaryDirectory(prefix="kinera-resume-r24-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "snapshot.json"
        self.inputs = ([0, 3], [0, 1], [[0.0] * 10, [1.0] + [0.0] * 9])

    def _model(self) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=len(self.characters),
            syllable_vocab_size=len(self.syllables),
            char_embedding_dim=3,
            syllable_embedding_dim=4,
            feature_dim=10,
            seed=17,
            backend=CPUBackend(),
        )

    def _weights(self, model: AutoregressiveModel) -> dict:
        rep = model.representation
        return {
            "char": copy.deepcopy(rep.char_embedding.weights),
            "syllable": copy.deepcopy(rep.syllable_embedding.weights),
            "heads": {
                name: {
                    "weights": copy.deepcopy(getattr(model.heads, name).weights),
                    "bias": copy.deepcopy(getattr(model.heads, name).bias),
                }
                for name in ("char", "syllable", "boundary")
            },
        }

    def _args(self, *, mode: str, learning_rate: float = 0.04, **overrides) -> dict:
        return {
            "examples": self.examples,
            "character_vocabulary": self.characters,
            "syllable_vocabulary": self.syllables,
            "mode": mode,
            "learning_rate": learning_rate,
            **overrides,
        }

    def test_multitask_continuous_4_equals_2_save_load_plus_2_exact(self) -> None:
        direct_model = self._model()
        loss = MultiTaskLoss(
            char_weight=1.0, syllable_weight=0.5, boundary_weight=0.2,
            backend=direct_model.backend,
        )
        direct = train_multitask_examples(
            direct_model, self.examples, self.characters,
            epochs=4, learning_rate=0.04, loss_function=loss,
        )
        split_model = self._model()
        split_loss = MultiTaskLoss(
            char_weight=1.0, syllable_weight=0.5, boundary_weight=0.2,
            backend=split_model.backend,
        )
        args = self._args(mode="multitask", loss_function=split_loss)
        first, state = train_hybrid_epochs(split_model, epochs=2, **args)
        self.assertEqual(first["epochs"], 2)
        self.assertEqual(state["global_step"], 6)
        before_save = self._weights(split_model)
        saved = save_hybrid_training_checkpoint(self.path, split_model, state, **args)
        self.assertEqual(saved, self.path)

        restored, restored_state, metadata = load_hybrid_training_checkpoint(
            self.path, **args
        )
        self.assertEqual(metadata, {})
        self.assertEqual(self._weights(restored), before_save)
        self.assertEqual(restored_state, state)
        self.assertIsNone(restored.last_logits)
        self.assertIsNone(restored.last_context_size)
        self.assertEqual(restored.forward_heads(*self.inputs), split_model.forward_heads(*self.inputs))

        resumed_loss = MultiTaskLoss(
            char_weight=1.0, syllable_weight=0.5, boundary_weight=0.2,
            backend=restored.backend,
        )
        final, final_state = train_hybrid_epochs(
            restored, epochs=2, previous_state=restored_state,
            **self._args(mode="multitask", loss_function=resumed_loss),
        )
        self.assertEqual(self._weights(restored), self._weights(direct_model))
        self.assertEqual(restored.forward_heads(*self.inputs), direct_model.forward_heads(*self.inputs))
        self.assertEqual(restored.predict(*self.inputs), direct_model.predict(*self.inputs))
        self.assertEqual(final["history"], direct["history"])
        self.assertEqual(final_state["history"], direct["history"])
        self.assertEqual(final["epochs"], 4)
        self.assertEqual(final["global_step"], 12)
        self.assertEqual(final_state["completed_epochs"], 4)
        self.assertEqual(restored_state["completed_epochs"], 2)

    def test_char_continuous_3_equals_1_save_load_plus_2_exact(self) -> None:
        direct_model = self._model()
        direct = train_examples(
            direct_model, self.examples, self.characters,
            epochs=3, learning_rate=0.07,
        )
        split_model = self._model()
        args = self._args(mode="char", learning_rate=0.07)
        _, state = train_hybrid_epochs(split_model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, split_model, state, **args)
        restored, old_state, _ = load_hybrid_training_checkpoint(self.path, **args)
        final, state2 = train_hybrid_epochs(
            restored, epochs=2, previous_state=old_state, **args
        )
        self.assertEqual(self._weights(restored), self._weights(direct_model))
        self.assertEqual(final["history"], direct["history"])
        self.assertEqual(state2["global_step"], 9)
        self.assertEqual(restored.forward_heads(*self.inputs), direct_model.forward_heads(*self.inputs))

    def test_rejects_data_config_mode_and_loss_changes_without_mutating_file(self) -> None:
        model = self._model()
        args = self._args(mode="multitask", max_examples=2)
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        original = self.path.read_bytes()
        changed = copy.deepcopy(self.examples)
        changed[0]["target_char_id"] = 3
        cases = [
            {"examples": changed},
            {"examples": list(reversed(self.examples))},
            {"learning_rate": 0.05},
            {"mode": "char"},
            {"max_examples": 3},
            {"loss_function": MultiTaskLoss(
                char_weight=0.7, backend=model.backend
            )},
        ]
        for overrides in cases:
            with self.subTest(case=str(overrides)[:130]):
                with self.assertRaisesRegex(CheckpointError, "incompatível"):
                    load_hybrid_training_checkpoint(
                        self.path, **{**args, **overrides}
                    )
                self.assertEqual(self.path.read_bytes(), original)

        changed_char = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "b", "a"]
        )
        with self.assertRaisesRegex(CheckpointError, "identidade"):
            load_hybrid_training_checkpoint(
                self.path, **{**args, "character_vocabulary": changed_char}
            )
        self.assertEqual(self.path.read_bytes(), original)

    def test_invalid_resume_state_and_corrupt_epoch_counters_fail_closed(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        original = self.path.read_bytes()
        for change in (
            {"completed_epochs": 3},
            {"global_step": 2},
            {"schema_version": 2},
            {"examples_sha256": "nope"},
            {"configuration": {**state["configuration"], "learning_rate": 0.5}},
            {"history": []},
        ):
            with self.subTest(change=change):
                payload = json.loads(original)
                payload["training"].update(change)
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(CheckpointError):
                    load_hybrid_training_checkpoint(self.path, **args)
        self.path.write_bytes(original)
        damaged = copy.deepcopy(state)
        damaged["global_step"] = 1
        with self.assertRaisesRegex(CheckpointError, "global_step"):
            save_hybrid_training_checkpoint(self.path, model, damaged, **args)
        self.assertEqual(self.path.read_bytes(), original)

    def test_inference_only_is_not_resumable_and_atomic_replace_failure(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        save_autoregressive_checkpoint(
            self.path, model,
            character_vocabulary=self.characters,
            syllable_vocabulary=self.syllables,
        )
        with self.assertRaisesRegex(CheckpointError, "inferência"):
            load_hybrid_training_checkpoint(self.path, **args)
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        before = self.path.read_bytes()
        with patch(
            "kinera.training.checkpoint.os.replace",
            side_effect=OSError("injected write interruption"),
        ):
            with self.assertRaisesRegex(CheckpointError, "salvar"):
                save_hybrid_training_checkpoint(self.path, model, state, **args)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".*.tmp")), [])

    def test_invalid_previous_state_rejected_before_training(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        before = self._weights(model)
        state["configuration"]["max_examples"] = 1
        with self.assertRaisesRegex(CheckpointError, "incompatível"):
            train_hybrid_epochs(model, epochs=2, previous_state=state, **args)
        self.assertEqual(before, self._weights(model))

    def test_inference_load_of_resumable_file_exposes_validated_metadata_only(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        restored, info = load_autoregressive_checkpoint(
            self.path,
            expected_character_vocabulary=self.characters,
            expected_syllable_vocabulary=self.syllables,
        )
        self.assertIsInstance(restored, AutoregressiveModel)
        self.assertEqual(info["training"], state)
        self.assertEqual(restored.forward_heads(*self.inputs), model.forward_heads(*self.inputs))


    def test_model_weights_must_match_saved_epoch_cursor(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        _, first_state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, first_state, **args)
        original = self.path.read_bytes()
        # Advance the live model without updating the older epoch snapshot.
        train_hybrid_epochs(model, epochs=1, previous_state=first_state, **args)
        with self.assertRaisesRegex(CheckpointError, "parâmetros"):
            save_hybrid_training_checkpoint(self.path, model, first_state, **args)
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaisesRegex(CheckpointError, "parâmetros"):
            train_hybrid_epochs(model, epochs=1, previous_state=first_state, **args)

    def test_finite_weight_tampering_rejected_for_training_snapshots(self) -> None:
        model = self._model()
        args = self._args(mode="multitask")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["model"]["parameters"]["heads"]["char"]["bias"][0] += 0.25
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        altered = self.path.read_bytes()
        with self.assertRaisesRegex(CheckpointError, "parâmetros"):
            load_hybrid_training_checkpoint(self.path, **args)
        self.assertEqual(self.path.read_bytes(), altered)
        # This binding applies to the R2.4 training record only. The R2.2
        # inference-only format still has no standalone weights checksum.



    def test_history_validation_rejects_invalid_char_rows_on_save_and_load(self) -> None:
        model = self._model()
        args = self._args(mode="char")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        baseline = self.path.read_bytes()
        invalid_rows = [
            {"loss": "inválida", "accuracy": 0.5},
            {"loss": 0.5, "accuracy": True},
            {"loss": [0.5], "accuracy": 0.5},
            {"loss": 0.5},
            {"loss": 0.5, "accuracy": 0.5, "unexpected": 0},
            {"loss": float("nan"), "accuracy": 0.5},
            {"loss": float("inf"), "accuracy": 0.5},
        ]
        for invalid in invalid_rows:
            with self.subTest(row=invalid):
                damaged = copy.deepcopy(state)
                damaged["history"][0] = invalid
                with self.assertRaises(CheckpointError):
                    save_hybrid_training_checkpoint(self.path, model, damaged, **args)
                self.assertEqual(self.path.read_bytes(), baseline)
                payload = json.loads(baseline)
                payload["training"] = damaged
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    with self.assertRaises(CheckpointError):
                        load_hybrid_training_checkpoint(self.path, **args)
                finally:
                    self.path.write_bytes(baseline)

    def test_history_validation_rejects_invalid_multitask_metrics(self) -> None:
        model = self._model()
        args = self._args(mode="multitask")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        baseline = self.path.read_bytes()
        self.assertIsNone(state["history"][0]["boundary_baseline_majority"])
        self.assertEqual(
            state["history"][0]["metric_results"]["boundary_baseline_majority"]["availability"],
            "unavailable",
        )
        changes = [
            ("scalar_string", ("char_loss",), "inválida"),
            ("scalar_boolean", ("boundary_accuracy",), True),
            ("distribution_string", ("char_target_distribution", "3"), "bad"),
            ("confusion_boolean", ("boundary_confusion", "tp"), True),
            ("metric_result_value_string", ("metric_results", "char_loss", "value"), "bad"),
            ("metric_result_wrong_shape", ("metric_results", "char_loss"), []),
            ("metric_result_bad_status", ("metric_results", "char_loss", "availability"), "broken"),
            ("metric_result_bool_support", ("metric_results", "char_loss", "support", "examples"), True),

            # Valid numeric types can still carry impossible rates and counts.
            ("char_accuracy_above_one", ("char_accuracy",), 5.0),
            ("boundary_accuracy_negative", ("boundary_accuracy",), -0.1),
            ("boundary_precision_above_one", ("boundary_precision",), 1.1),
            ("boundary_recall_negative", ("boundary_recall",), -0.2),
            ("boundary_f1_above_one", ("boundary_f1",), 2.0),
            ("boundary_specificity_above_one", ("boundary_specificity",), 1.1),
            ("boundary_baseline_negative", ("boundary_baseline_constant_0",), -0.1),
            ("nested_char_accuracy_above_one", ("metric_results", "char_accuracy", "value"), 5.0),
            ("nested_boundary_f1_above_one", ("metric_results", "boundary_f1", "value"), 5.0),
            ("nested_true_negative_over_sample_count", ("metric_results", "boundary_true_negative", "value"), 99),
            ("nested_examples_support_mismatch", ("metric_results", "char_loss", "support", "examples"), 0),
            ("true_negative_over_sample_count", ("boundary_true_negative",), 99),
            # Valid nonnegative integers can contradict the example count
            # and the binary confusion matrix.
            ("char_target_sum_mismatch", ("char_target_distribution", "3"), 99),
            ("syllable_target_sum_mismatch", ("syllable_target_distribution", "0"), 99),
            ("boundary_target_count_mismatch", ("boundary_target_distribution", "1"), 99),
            ("boundary_prediction_count_mismatch", ("boundary_prediction_distribution", "0"), 99),
            ("target_status_sum_mismatch", ("syllable_target_status_distribution", "known"), 99),
            ("confusion_total_mismatch", ("boundary_confusion", "tp"), 99),
        ]
        for name, path, replacement in changes:
            with self.subTest(name=name):
                damaged = copy.deepcopy(state)
                target = damaged["history"][0]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(CheckpointError):
                    save_hybrid_training_checkpoint(self.path, model, damaged, **args)
                self.assertEqual(self.path.read_bytes(), baseline)
                payload = json.loads(baseline)
                payload["training"] = damaged
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    with self.assertRaises(CheckpointError):
                        load_hybrid_training_checkpoint(self.path, **args)
                finally:
                    self.path.write_bytes(baseline)


    def test_history_rejects_binary_count_conflicts_with_valid_totals(self) -> None:
        model = self._model()
        args = self._args(mode="multitask")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        original = self.path.read_bytes()
        total = len(self.examples)

        modified_target = copy.deepcopy(state)
        target_dist = modified_target["history"][0]["boundary_target_distribution"]
        target_dist["0"], target_dist["1"] = target_dist["1"], target_dist["0"]

        modified_prediction = copy.deepcopy(state)
        prediction_dist = modified_prediction["history"][0]["boundary_prediction_distribution"]
        prediction_dist["0"] = (prediction_dist["0"] + 1) % (total + 1)
        prediction_dist["1"] = total - prediction_dist["0"]

        modified_confusion = copy.deepcopy(state)
        matrix = modified_confusion["history"][0]["boundary_confusion"]
        source = next(key for key in ("tp", "tn", "fp", "fn") if matrix[key] > 0)
        destination = next(key for key in ("tp", "tn", "fp", "fn") if key != source)
        matrix[source] -= 1
        matrix[destination] += 1

        modified_true_negative = copy.deepcopy(state)
        row = modified_true_negative["history"][0]
        row["boundary_true_negative"] = (
            0 if row["boundary_confusion"]["tn"] == total
            else row["boundary_confusion"]["tn"] + 1
        )

        for name, damaged in (
            ("target_distribution", modified_target),
            ("prediction_distribution", modified_prediction),
            ("confusion_matrix", modified_confusion),
            ("true_negative", modified_true_negative),
        ):
            with self.subTest(name=name):
                with self.assertRaises(CheckpointError):
                    save_hybrid_training_checkpoint(self.path, model, damaged, **args)
                self.assertEqual(self.path.read_bytes(), original)
                payload = json.loads(original)
                payload["training"] = damaged
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    with self.assertRaises(CheckpointError):
                        load_hybrid_training_checkpoint(self.path, **args)
                finally:
                    self.path.write_bytes(original)


    def test_history_cross_field_consistency_rejects_contradictory_values(self) -> None:
        model = self._model()
        args = self._args(mode="multitask")
        _, state = train_hybrid_epochs(model, epochs=1, **args)
        save_hybrid_training_checkpoint(self.path, model, state, **args)
        original = self.path.read_bytes()
        row = state["history"][0]

        def alternate(value: float) -> float:
            return 0.0 if value != 0.0 else 1.0

        # Each replacement stays finite and within its permitted range.
        # Save/load must not accept values that disagree with the same row.
        cases = (
            ("precision_vs_confusion", ("boundary_precision",),
             alternate(row["boundary_precision"])),
            ("recall_vs_confusion", ("boundary_recall",),
             alternate(row["boundary_recall"])),
            ("f1_vs_confusion", ("boundary_f1",),
             alternate(row["boundary_f1"])),
            ("accuracy_vs_confusion", ("boundary_accuracy",),
             alternate(row["boundary_accuracy"])),
            ("constant_baseline_vs_targets", ("boundary_baseline_constant_0",),
             alternate(row["boundary_baseline_constant_0"])),
            ("nested_char_accuracy_vs_scalar",
             ("metric_results", "char_accuracy", "value"),
             alternate(row["char_accuracy"])),
            ("nested_char_loss_vs_scalar",
             ("metric_results", "char_loss", "value"),
             row["char_loss"] + 0.125),
            ("nested_true_negative_vs_confusion",
             ("metric_results", "boundary_true_negative", "value"),
             row["boundary_true_negative"] + 1 if row["boundary_true_negative"] < len(self.examples)
             else row["boundary_true_negative"] - 1),
            ("nested_boundary_accuracy_vs_confusion",
             ("metric_results", "boundary_accuracy", "value"),
             alternate(row["boundary_accuracy"])),
        )
        for name, path, replacement in cases:
            with self.subTest(name=name):
                damaged = copy.deepcopy(state)
                target = damaged["history"][0]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(CheckpointError):
                    save_hybrid_training_checkpoint(self.path, model, damaged, **args)
                self.assertEqual(self.path.read_bytes(), original)
                payload = json.loads(original)
                payload["training"] = damaged
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                try:
                    with self.assertRaises(CheckpointError):
                        load_hybrid_training_checkpoint(self.path, **args)
                finally:
                    self.path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
