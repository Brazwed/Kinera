"""Regression checks for the installed Kinera package layout."""

import ast
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import os
import subprocess
import sys
import unittest


class PackageLayoutTests(unittest.TestCase):
    def test_kinera_is_importable_as_a_package(self) -> None:
        self.assertIsNotNone(
            find_spec("kinera"),
            "installable source package 'kinera' is missing",
        )

    def test_existing_runtime_modules_are_importable_under_kinera(self) -> None:
        for module_name in (
            "kinera.nn.linear",
            "kinera.language.dataset",
            "kinera.training.autoregressive",
            "kinera.scripts.build_character_dataset",
        ):
            with self.subTest(module=module_name):
                try:
                    spec = find_spec(module_name)
                except ModuleNotFoundError:
                    spec = None
                self.assertIsNotNone(spec, f"{module_name} is missing from src/kinera")


    def test_dataset_cli_compatibility_wrapper_remains_runnable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "scripts.build_character_dataset", "--help"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--annotated", result.stdout)


    def test_new_cli_entrypoints_are_runnable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        commands = (
            ("scripts.train", "--train"),
            ("scripts.compare", "--variants"),
            ("scripts.prepare_dataset", "--annotated"),
        )
        for module_name, expected_option in commands:
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [sys.executable, "-m", module_name, "--help"],
                    cwd=project_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected_option, result.stdout)

    def test_target_architecture_modules_are_importable(self) -> None:
        module_names = (
            "kinera.nn.layers.linear",
            "kinera.nn.layers.embedding",
            "kinera.nn.heads.character",
            "kinera.nn.heads.syllable",
            "kinera.nn.heads.boundary",
            "kinera.nn.losses.cross_entropy",
            "kinera.nn.losses.binary_cross_entropy",
            "kinera.nn.losses.multitask",
            "kinera.backends.base",
            "kinera.backends.cpu.backend",
            "kinera.backends.cpu.ops.linear",
            "kinera.backends.cpu.ops.embedding",
            "kinera.backends.cpu.ops.activations",
            "kinera.backends.cpu.ops.probability",
            "kinera.backends.cuda.backend",
            "kinera.backends.cuda.runtime",
            "kinera.backends.cuda.memory",
            "kinera.backends.cuda.bridge",
            "kinera.nn.models.hybrid",
            "kinera.monitor",
            "kinera.monitor.metrics",
            "kinera.monitor.telemetry",
            "kinera.monitor.telemetry.records",
            "kinera.monitor.telemetry.timing",
            "kinera.language.orthography.alignment",
            "kinera.language.orthography.features",
            "kinera.language.syllables.tokenizer",
            "kinera.language.syllables.causal",
            "kinera.language.vocabulary.character",
            "kinera.language.vocabulary.syllable",
            "kinera.language.datasets.records",
            "kinera.language.datasets.loader",
            "kinera.language.datasets.builder",
            "kinera.training.trainer",
            "kinera.training.evaluator",
            "kinera.training.checkpoint",
            "kinera.training.config",
            "kinera.experiments.runner",
            "kinera.experiments.synthetic_data",
            "kinera.experiments.comparison",
            "kinera.experiments.multi_seed",
            "kinera.experiments.variants",
            "kinera.diagnostics.gradients",
            "kinera.diagnostics.syllables",
            "kinera.diagnostics.orthography",
        )
        for module_name in module_names:
            with self.subTest(module=module_name):
                try:
                    spec = find_spec(module_name)
                except ModuleNotFoundError:
                    spec = None
                self.assertIsNotNone(spec, f"{module_name} is missing from the target layout")
                import_module(module_name)

    def test_telemetry_import_is_cpu_only_and_does_not_call_clocks(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "PYTHONPATH": str(project_root / "src"),
        }
        script = (
            "import sys, time; "
            "time.monotonic_ns = lambda: (_ for _ in ()).throw(AssertionError('clock called')); "
            "import kinera.monitor; import kinera.monitor.telemetry; "
            "import kinera.monitor.telemetry.records; "
            "import kinera.monitor.telemetry.timing; "
            "assert not any(name.startswith('kinera.backends.cuda') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_compatibility_entrypoints_are_runnable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        commands = (
            ("run_autoregressive.py", "--vocabulary"),
            ("run_comparison.py", "--variants"),
            ("run_experiment.py", "--seeds"),
            ("run_syllable_analysis.py", "--top-examples"),
        )
        for script_name, expected_option in commands:
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, script_name, "--help"],
                    cwd=project_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected_option, result.stdout)

    def test_nn_does_not_import_concrete_backend_modules(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        nn_root = project_root / "src" / "kinera" / "nn"
        forbidden_prefixes = ("kinera.backends.cpu", "kinera.backends.cuda")
        for source_path in nn_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)
                elif isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
            for module_name in imported_modules:
                with self.subTest(path=source_path, module=module_name):
                    self.assertFalse(
                        module_name.startswith(forbidden_prefixes),
                        f"nn imports concrete backend {module_name}",
                    )


if __name__ == "__main__":
    unittest.main()
