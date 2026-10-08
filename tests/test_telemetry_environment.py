import unittest
from pathlib import Path
from unittest.mock import patch

from kinera.monitor.telemetry.environment import collect_environment_snapshot


UTC = "2026-10-07T12:00:00+00:00"


class EnvironmentCollectorTests(unittest.TestCase):
    def test_explicit_collection_uses_stdlib_values_and_linux_cpuinfo_fixture(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.python_version", return_value="3.10.14"),
            patch("kinera.monitor.telemetry.environment.sys.executable", "/opt/python/bin/python"),
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Linux"),
            patch("kinera.monitor.telemetry.environment.platform.version", return_value="build-tag"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, return_value={"VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04 LTS"}),
            patch("kinera.monitor.telemetry.environment.platform.release", return_value="6.8.0-fixture"),
            patch("kinera.monitor.telemetry.environment.platform.machine", return_value="x86_64"),
            patch("kinera.monitor.telemetry.environment.os.cpu_count", return_value=8),
            patch("kinera.monitor.telemetry.environment.os.sched_getaffinity", return_value={0, 2, 3}),
        ):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                proc_root=Path("/fixture/proc"),
                read_text=lambda path: "processor : 0\nmodel name : Fixture CPU 9000\n",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        encoded = snapshot.to_dict()
        self.assertEqual(encoded["environment_schema_version"], 1)
        self.assertEqual(encoded["captured_at"], UTC)
        self.assertEqual(encoded["python_version"], "3.10.14")
        self.assertEqual(encoded["python_executable"], "/opt/python/bin/python")
        self.assertEqual(encoded["os_system"], "Linux")
        self.assertEqual(encoded["os_release"], "24.04")
        self.assertEqual(encoded["platform_version"], "build-tag")
        self.assertEqual(encoded["kernel_release"], "6.8.0-fixture")
        self.assertEqual(encoded["architecture"], "x86_64")
        self.assertEqual(encoded["cpu_model"], "Fixture CPU 9000")
        self.assertEqual(encoded["logical_cpu_count"], 8)
        self.assertEqual(encoded["cpu_affinity"], [0, 2, 3])
        self.assertNotEqual(encoded["logical_cpu_count"], len(encoded["cpu_affinity"]))
        self.assertEqual(encoded["sources"]["cpu_model"], "/proc/cpuinfo:model name")
        self.assertEqual(
            encoded["sources"]["os_release"],
            "platform.freedesktop_os_release:VERSION_ID",
        )
        self.assertEqual(encoded["sources"]["platform_version"], "platform.version")
    def test_os_release_uses_distribution_metadata_not_platform_version(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.python_version", return_value="3.14.7"),
            patch("kinera.monitor.telemetry.environment.sys.executable", "/fixture/python"),
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Linux"),
            patch("kinera.monitor.telemetry.environment.platform.version", return_value="kernel-build-tag"),
            patch("kinera.monitor.telemetry.environment.platform.release", return_value="6.8.0-fixture"),
            patch("kinera.monitor.telemetry.environment.platform.machine", return_value="x86_64"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, return_value={
                "VERSION_ID": "24.04",
                "PRETTY_NAME": "Ubuntu 24.04 LTS",
            }),
            patch("kinera.monitor.telemetry.environment.os.cpu_count", return_value=4),
            patch("kinera.monitor.telemetry.environment.os.sched_getaffinity", return_value={0, 1}),
        ):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=lambda path: "model name: fixture CPU\n",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.os_release, "24.04")
        self.assertEqual(snapshot.platform_version, "kernel-build-tag")
        self.assertEqual(snapshot.sources["os_release"], "platform.freedesktop_os_release:VERSION_ID")
        self.assertEqual(snapshot.sources["platform_version"], "platform.version")

    def test_unavailable_sources_are_not_encoded_as_zero_or_scientific_failure(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.python_version", return_value="3.10.14"),
            patch("kinera.monitor.telemetry.environment.sys.executable", ""),
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Linux"),
            patch("kinera.monitor.telemetry.environment.platform.version", return_value="build-tag"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, side_effect=PermissionError),
            patch("kinera.monitor.telemetry.environment.platform.release", return_value="6.8.0-fixture"),
            patch("kinera.monitor.telemetry.environment.platform.machine", return_value="x86_64"),
            patch("kinera.monitor.telemetry.environment.os.cpu_count", return_value=None),
            patch("kinera.monitor.telemetry.environment.os.sched_getaffinity", side_effect=PermissionError),
        ):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=lambda path: "not a cpu model record\n",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        encoded = snapshot.to_dict()
        self.assertIsNone(encoded["python_executable"])
        self.assertIsNone(encoded["logical_cpu_count"])
        self.assertIsNone(encoded["cpu_affinity"])
        self.assertIsNone(encoded["cpu_model"])
        self.assertIsNone(encoded["os_release"])
        self.assertEqual(encoded["platform_version"], "build-tag")
        self.assertEqual(encoded["unavailable"]["python_executable"], "field missing or empty")
        self.assertEqual(encoded["unavailable"]["logical_cpu_count"], "os.cpu_count returned no positive count")
        self.assertIn("PermissionError", encoded["unavailable"]["cpu_affinity"])
        self.assertIn("PermissionError", encoded["unavailable"]["os_release"])
        self.assertIn("cpu_model", encoded["unavailable"])

    def test_os_release_falls_back_to_pretty_name_when_version_id_missing(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Linux"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, return_value={"PRETTY_NAME": "Alpine Linux v3.22"}),
        ):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=lambda path: "model name: fixture CPU\n",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.os_release, "Alpine Linux v3.22")
        self.assertEqual(snapshot.sources["os_release"], "platform.freedesktop_os_release:PRETTY_NAME")

    def test_os_release_is_not_applicable_off_linux(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.python_version", return_value="3.14.7"),
            patch("kinera.monitor.telemetry.environment.sys.executable", "/fixture/python"),
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Windows"),
            patch("kinera.monitor.telemetry.environment.platform.version", return_value="win-build"),
            patch("kinera.monitor.telemetry.environment.platform.release", return_value="11"),
            patch("kinera.monitor.telemetry.environment.platform.machine", return_value="AMD64"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, side_effect=AssertionError),
            patch("kinera.monitor.telemetry.environment.os.cpu_count", return_value=4),
            patch("kinera.monitor.telemetry.environment.os.sched_getaffinity", None),
        ):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=lambda path: (_ for _ in ()).throw(AssertionError("Linux procfs touched")),
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.os_release)
        self.assertEqual(snapshot.platform_version, "win-build")
        self.assertIn("os_release", snapshot.not_applicable)
        self.assertIn("cpu_model", snapshot.not_applicable)

    def test_unknown_os_does_not_mislabel_cpu_model_as_not_applicable(self) -> None:
        with patch("kinera.monitor.telemetry.environment.platform.system", side_effect=OSError):
            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=lambda path: "model name: fixture CPU\n",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.os_system)
        self.assertIn("cpu_model", snapshot.unavailable)
        self.assertNotIn("cpu_model", snapshot.not_applicable)

    def test_missing_cpuinfo_is_partial_not_an_exception(self) -> None:
        with (
            patch("kinera.monitor.telemetry.environment.platform.system", return_value="Linux"),
            patch("kinera.monitor.telemetry.environment.platform.freedesktop_os_release", create=True, return_value={"VERSION_ID": "fixture-release"}),
            patch("kinera.monitor.telemetry.environment.os.sched_getaffinity", None),
        ):
            def missing_file(path: Path) -> str:
                raise FileNotFoundError(path)

            snapshot = collect_environment_snapshot(
                captured_at=UTC,
                read_text=missing_file,
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.cpu_model)
        self.assertIn("FileNotFoundError", snapshot.unavailable["cpu_model"])
        self.assertIn("cpu_affinity", snapshot.not_applicable)


if __name__ == "__main__":
    unittest.main()
