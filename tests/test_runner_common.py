import json
import tempfile
import unittest
from pathlib import Path

from training.runner_common import (
    load_seed_checkpoint,
    write_json_checkpoint,
)


class RunnerCommonTests(unittest.TestCase):
    def test_checkpoint_write_is_atomic_and_resume_validates_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.json"
            report = {"phase": "test", "seeds": [7], "runs": [], "status": "running"}
            write_json_checkpoint(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            loaded = load_seed_checkpoint(
                path,
                {"phase": "test"},
                [7, 8],
            )
            self.assertEqual(loaded["runs"], [])

    def test_resume_rejects_incompatible_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_json_checkpoint(path, {"phase": "a", "seeds": [7], "runs": []})
            with self.assertRaises(ValueError):
                load_seed_checkpoint(path, {"phase": "b"}, [7])

    def test_workers_are_operational_and_do_not_block_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_json_checkpoint(
                path,
                {"phase": "test", "workers": 3, "seeds": [7], "runs": []},
            )

            loaded = load_seed_checkpoint(path, {"phase": "test"}, [7, 42])

        self.assertEqual(loaded["workers"], 3)


if __name__ == "__main__":
    unittest.main()
