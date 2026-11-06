import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.legacy import phase5_3, phase5_5
from kinera.experiments.config import ExperimentConfig
from kinera.experiments.fingerprint import build_legacy_scientific_configuration
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


class LegacyCheckpointIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.characters_path = self.root / "characters.json"
        self.syllables_path = self.root / "syllables.json"
        self.train_path = self.root / "train.jsonl"
        self.validation_path = self.root / "validation.jsonl"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "a", "b"]
        )
        self.syllables = SyllableVocabulary(["<UNK_SYL>", "ab"])
        self.characters_path.write_text(
            json.dumps(self.characters.to_dict()), encoding="utf-8"
        )
        self.syllables_path.write_text(
            json.dumps(self.syllables.to_dict()), encoding="utf-8"
        )
        self.train_record = {
            "word_key": "ab",
            "target_char": "a",
            "context_ids": [0, 3],
            "context_syllable_ids": [0, 1],
            "context_feature_vectors": [[0.0] * 10, [1.0] * 10],
            "target_char_id": 3,
            "target_syllable_id": 1,
            "target_is_syllable_end": True,
        }
        self.validation_record = dict(self.train_record, target_char_id=4, target_char="b")
        self._write_datasets()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_datasets(self) -> None:
        self.train_path.write_text(
            json.dumps(self.train_record) + "\n", encoding="utf-8"
        )
        self.validation_path.write_text(
            json.dumps(self.validation_record) + "\n", encoding="utf-8"
        )

    def _arguments(
        self,
        *,
        phase: str,
        seeds: tuple[int, ...],
        resume: bool,
        workers: int = 1,
        epochs: int = 1,
        train_path: Path | None = None,
        validation_path: Path | None = None,
    ) -> list[str]:
        module = phase5_3 if phase == "5.3" else phase5_5
        args = [
            module.__file__,
            "--vocabulary", str(self.characters_path),
            "--syllable-vocabulary", str(self.syllables_path),
            "--train", str(self.train_path if train_path is None else train_path),
            "--validation", str(
                self.validation_path if validation_path is None else validation_path
            ),
            "--seeds", *map(str, seeds),
            "--epochs", str(epochs),
            "--learning-rate", "0.05",
            "--max-examples", "10",
            "--workers", str(workers),
            "--checkpoint", str(self.checkpoint_path),
        ]
        if phase == "5.3":
            args.extend(["--top-examples", "2"])
        if resume:
            args.append("--resume")
        return args

    def _fake_diagnostic(self, calls: list[int]):
        def phase53(*args, **kwargs):
            seed = kwargs["seeds"][0]
            calls.append(seed)
            pair = {
                "mean_delta_loss": 0.0,
                "accuracy_delta": 0.0,
                "improved_loss_fraction": 0.0,
                "regressed_loss_fraction": 0.0,
            }
            return {
                "runs": [{
                    "seed": seed,
                    "pair_deltas": {
                        "A_vs_C": {"overall": pair},
                        "B_vs_D": {"overall": pair},
                    },
                }]
            }

        def phase55(*args, **kwargs):
            seed = kwargs["seeds"][0]
            calls.append(seed)
            return {"runs": [{"seed": seed, "comparisons": {}}]}

        return phase53, phase55

    def _invoke(
        self,
        phase: str,
        *,
        seeds=(7,),
        resume=False,
        calls=None,
        workers=1,
        epochs=1,
        train_path=None,
        validation_path=None,
    ) -> None:
        module = phase5_3 if phase == "5.3" else phase5_5
        phase53_fake, phase55_fake = self._fake_diagnostic(calls if calls is not None else [])
        diagnostic_name = (
            "run_syllable_gain_analysis"
            if phase == "5.3"
            else "run_orthographic_class_analysis"
        )
        diagnostic = phase53_fake if phase == "5.3" else phase55_fake
        with (
            mock.patch(
                "sys.argv",
                self._arguments(
                    phase=phase,
                    seeds=seeds,
                    resume=resume,
                    workers=workers,
                    epochs=epochs,
                    train_path=train_path,
                    validation_path=validation_path,
                ),
            ),
            mock.patch.object(module, diagnostic_name, diagnostic),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            module.main()

    def test_both_phases_fail_closed_on_legacy_checkpoints_without_mutation(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.write_bytes(
                    b'{"seeds":[7],"runs":[{"seed":7,"status":"completed"}]}\n'
                )
                before = self.checkpoint_path.read_bytes()
                calls: list[int] = []
                with self.assertRaisesRegex(ValueError, "sem identidade científica"):
                    self._invoke(phase, resume=True, calls=calls)
                self.assertEqual(calls, [])
                self.assertEqual(self.checkpoint_path.read_bytes(), before)

    def test_both_phases_resume_compatible_seeds_and_allow_seed_expansion(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                calls: list[int] = []
                self._invoke(phase, seeds=(7,), calls=calls)
                initial = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self.assertIn("experiment_identity", initial)
                self.assertEqual(calls, [7])

                calls.clear()
                self._invoke(phase, seeds=(7, 42), resume=True, calls=calls)
                resumed = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self.assertEqual(calls, [42])
                self.assertEqual(resumed["completed_runs"], 2)
                self.assertEqual(
                    [run["seed"] for run in resumed["runs"]],
                    [7, 42],
                )

    def test_data_change_rejected_before_execution_and_checkpoint_write(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                self._invoke(phase)
                self.train_record["target_char_id"] = 4
                self._write_datasets()
                before = self.checkpoint_path.read_bytes()
                calls: list[int] = []
                with self.assertRaisesRegex(ValueError, "training_examples_sha256"):
                    self._invoke(phase, resume=True, calls=calls)
                self.assertEqual(calls, [])
                self.assertEqual(self.checkpoint_path.read_bytes(), before)
                self.train_record["target_char_id"] = 3
                self._write_datasets()

    def test_vocabulary_remapping_rejected_before_execution_and_checkpoint_write(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                self._invoke(phase)
                before = self.checkpoint_path.read_bytes()
                reordered = CharacterVocabulary(
                    ["<START>", "<END>", "<UNK>", "b", "a"]
                )
                self.characters_path.write_text(
                    json.dumps(reordered.to_dict()), encoding="utf-8"
                )
                calls: list[int] = []
                with self.assertRaisesRegex(
                    ValueError, "character_vocabulary_sha256"
                ):
                    self._invoke(phase, resume=True, calls=calls)
                self.assertEqual(calls, [])
                self.assertEqual(self.checkpoint_path.read_bytes(), before)
                self.characters_path.write_text(
                    json.dumps(self.characters.to_dict()), encoding="utf-8"
                )

    def test_scientific_configuration_change_rejected_before_execution(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                self._invoke(phase)
                before = self.checkpoint_path.read_bytes()
                calls: list[int] = []
                with self.assertRaisesRegex(
                    ValueError, "scientific_configuration_sha256"
                ):
                    self._invoke(phase, resume=True, calls=calls, epochs=2)
                self.assertEqual(calls, [])
                self.assertEqual(self.checkpoint_path.read_bytes(), before)

    def test_equal_content_at_new_paths_and_worker_change_remain_compatible(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                calls: list[int] = []
                self._invoke(phase, calls=calls)
                alternate_train = self.root / "copied-train.jsonl"
                alternate_validation = self.root / "copied-validation.jsonl"
                alternate_train.write_bytes(self.train_path.read_bytes())
                alternate_validation.write_bytes(self.validation_path.read_bytes())
                calls.clear()
                self._invoke(
                    phase,
                    resume=True,
                    calls=calls,
                    workers=2,
                    train_path=alternate_train,
                    validation_path=alternate_validation,
                )
                report = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self.assertEqual(calls, [])
                self.assertEqual(report["workers"], 2)
                self.assertEqual(report["completed_runs"], 1)

    def test_checkpoint_with_unknown_version_or_bad_digest_is_unchanged(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                self._invoke(phase)
                good_report = json.loads(
                    self.checkpoint_path.read_text(encoding="utf-8")
                )
                for mutation in ("version", "digest"):
                    bad_report = json.loads(json.dumps(good_report))
                    if mutation == "version":
                        bad_report["experiment_identity"][
                            "fingerprint_schema_version"
                        ] = 99
                    else:
                        bad_report["experiment_identity"]["digest"] = "0" * 64
                    self.checkpoint_path.write_text(
                        json.dumps(bad_report), encoding="utf-8"
                    )
                    before = self.checkpoint_path.read_bytes()
                    with self.assertRaises(ValueError):
                        self._invoke(phase, resume=True)
                    self.assertEqual(self.checkpoint_path.read_bytes(), before)
                self.checkpoint_path.write_text(
                    json.dumps(good_report), encoding="utf-8"
                )

    def test_worker_rechecks_identity_before_diagnostic_execution(self) -> None:
        for phase in ("5.3", "5.5"):
            with self.subTest(phase=phase):
                self.checkpoint_path.unlink(missing_ok=True)
                self._invoke(phase)
                report = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self.train_record["target_char_id"] = 4
                self._write_datasets()
                calls: list[int] = []
                phase53_fake, phase55_fake = self._fake_diagnostic(calls)
                module = phase5_3 if phase == "5.3" else phase5_5
                worker = module._run_seed if phase == "5.3" else module.run_seed
                experiment_configuration = ExperimentConfig(
                    phase=(
                        "5.3_expanded_analysis"
                        if phase == "5.3"
                        else "5.5_orthographic_class_analysis"
                    ),
                    variants=tuple("ABCD"),
                    seeds=(7,),
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=10,
                    top_examples=2 if phase == "5.3" else None,
                ).to_dict()
                scientific_configuration = build_legacy_scientific_configuration(
                    experiment_configuration,
                    vocabulary=self.characters,
                    syllable_vocabulary=self.syllables,
                )
                payload = {
                    "vocabulary": str(self.characters_path),
                    "syllable_vocabulary": str(self.syllables_path),
                    "train": str(self.train_path),
                    "validation": str(self.validation_path),
                    "max_examples": 10,
                    "scientific_configuration": scientific_configuration,
                    "experiment_identity": report["experiment_identity"],
                    "seed": 7,
                    "epochs": 1,
                    "learning_rate": 0.05,
                    "top_examples": 2,
                }
                with mock.patch.object(
                    module,
                    "run_syllable_gain_analysis"
                    if phase == "5.3"
                    else "run_orthographic_class_analysis",
                    phase53_fake if phase == "5.3" else phase55_fake,
                ):
                    result = worker(payload)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(calls, [])
                self.train_record["target_char_id"] = 3
                self._write_datasets()

if __name__ == "__main__":
    unittest.main()
