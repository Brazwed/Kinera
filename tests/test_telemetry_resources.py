import tempfile
import unittest
from pathlib import Path

from kinera.monitor.telemetry.resources import collect_resource_snapshot


UTC = "2026-10-07T12:00:00+00:00"


class ResourceCollectorTests(unittest.TestCase):
    def test_cgroup_v2_prefers_most_specific_matching_mount_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            (self_dir / "status").write_text("VmRSS:\t5 kB\n", encoding="utf-8")
            (self_dir / "cgroup").write_text("0::/tenant/job\n", encoding="utf-8")
            broad_mount = root / "broad"
            specific_mount = root / "specific"
            broad_group = broad_mount / "tenant" / "job"
            specific_group = specific_mount / "job"
            broad_group.mkdir(parents=True)
            specific_group.mkdir(parents=True)
            (broad_group / "memory.current").write_text("111\n", encoding="utf-8")
            (broad_group / "memory.peak").write_text("112\n", encoding="utf-8")
            (broad_group / "memory.max").write_text("113\n", encoding="utf-8")
            (specific_group / "memory.current").write_text("221\n", encoding="utf-8")
            (specific_group / "memory.peak").write_text("222\n", encoding="utf-8")
            (specific_group / "memory.max").write_text("223\n", encoding="utf-8")
            (self_dir / "mountinfo").write_text(
                f"29 23 0:26 / {broad_mount} rw - cgroup2 cgroup rw\n"
                f"30 23 0:27 /tenant {specific_mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.cgroup_memory_current_bytes, 221)
        self.assertEqual(snapshot.cgroup_memory_peak_bytes, 222)
        self.assertEqual(snapshot.cgroup_memory_limit_bytes, 223)

    def test_cgroup_v2_reads_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t5 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("0::/workload\n", encoding="utf-8")
            mount = root / "cgroup-mount"
            group = mount / "workload"
            group.mkdir(parents=True)
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {mount} rw,nosuid,nodev - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            (group / "memory.current").write_text("4096\n", encoding="utf-8")
            (group / "memory.peak").write_text("8192\n", encoding="utf-8")
            (group / "memory.max").write_text("16384\n", encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 12,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        encoded = snapshot.to_dict()
        self.assertEqual(encoded["cgroup_version"], "v2")
        self.assertEqual(encoded["cgroup_memory_current_bytes"], 4096)
        self.assertEqual(encoded["cgroup_memory_peak_bytes"], 8192)
        self.assertEqual(encoded["cgroup_memory_limit_bytes"], 16384)
        self.assertIs(encoded["cgroup_memory_limit_unlimited"], False)
        self.assertEqual(
            encoded["sources"]["cgroup_memory_current_bytes"],
            "cgroup-v2:memory.current",
        )

    def test_mountinfo_escaped_spaces_are_decoded_before_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t2 cgroup ") as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t3 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("0::/job\n", encoding="utf-8")
            mount = root / "cgroup mount"
            group = mount / "job"
            group.mkdir(parents=True)
            escaped_mount = str(mount).replace(" ", r"\040")
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {escaped_mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            (group / "memory.current").write_text("7\n", encoding="utf-8")
            (group / "memory.peak").write_text("8\n", encoding="utf-8")
            (group / "memory.max").write_text("9\n", encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.cgroup_memory_current_bytes, 7)
        self.assertEqual(snapshot.cgroup_memory_peak_bytes, 8)
        self.assertEqual(snapshot.cgroup_memory_limit_bytes, 9)

    def test_cgroup_v2_max_limit_and_missing_peak_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t0 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("0::/job\n", encoding="utf-8")
            mount = root / "cgroup-mount"
            group = mount / "job"
            group.mkdir(parents=True)
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            (group / "memory.current").write_text("0\n", encoding="utf-8")
            (group / "memory.max").write_text("max\n", encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 0,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        encoded = snapshot.to_dict()
        self.assertEqual(encoded["cgroup_memory_current_bytes"], 0)
        self.assertIsNone(encoded["cgroup_memory_peak_bytes"])
        self.assertIn("cgroup_memory_peak_bytes", encoded["unavailable"])
        self.assertIsNone(encoded["cgroup_memory_limit_bytes"])
        self.assertIs(encoded["cgroup_memory_limit_unlimited"], True)
        self.assertIn("cgroup_memory_limit_bytes", encoded["not_applicable"])

    def test_cgroup_v1_is_detected_but_metrics_are_partial_in_t2a(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t8 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("5:memory:/docker/job\n", encoding="utf-8")
            mount = root / "memory-mount"
            mount.mkdir()
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {mount} rw - cgroup cgroup rw,memory\n",
                encoding="utf-8",
            )

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 8,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.cgroup_version, "v1")
        self.assertIsNone(snapshot.cgroup_memory_current_bytes)
        self.assertIn("cgroup v1 memory collection is not implemented in T2a", snapshot.unavailable["cgroup_memory_current_bytes"])

    def test_cgroup_membership_path_traversal_is_rejected_before_mount_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            (self_dir / "status").write_text("VmRSS:\t5 kB\n", encoding="utf-8")
            (self_dir / "cgroup").write_text("0::/../../outside/job\n", encoding="utf-8")
            mount = root / "mount"
            outside = root / "outside" / "job"
            outside.mkdir(parents=True)
            (outside / "memory.current").write_text("123\n", encoding="utf-8")
            (outside / "memory.peak").write_text("456\n", encoding="utf-8")
            (outside / "memory.max").write_text("789\n", encoding="utf-8")
            (self_dir / "mountinfo").write_text(
                f"29 23 0:26 / {mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            opened: list[Path] = []

            def reader(path: Path) -> str:
                opened.append(path)
                return path.read_text(encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=reader,
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.cgroup_version)
        self.assertNotIn(self_dir / "mountinfo", opened)
        self.assertFalse(any(path.name == "memory.current" for path in opened))

    def test_cgroup_metric_symlink_cannot_escape_discovered_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            (self_dir / "status").write_text("VmRSS:\t2 kB\n", encoding="utf-8")
            (self_dir / "cgroup").write_text("0::/job\n", encoding="utf-8")
            mount = root / "cgroup-mount"
            group = mount / "job"
            group.mkdir(parents=True)
            outside = root / "outside-current"
            outside.write_text("999\n", encoding="utf-8")
            (group / "memory.current").symlink_to(outside)
            (group / "memory.peak").write_text("20\n", encoding="utf-8")
            (group / "memory.max").write_text("30\n", encoding="utf-8")
            (self_dir / "mountinfo").write_text(
                f"29 23 0:26 / {mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            opened: list[Path] = []

            def reader(path: Path) -> str:
                opened.append(path)
                return path.read_text(encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=reader,
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.cgroup_memory_current_bytes)
        self.assertIn("cgroup_memory_current_bytes", snapshot.unavailable)
        self.assertNotIn(outside, opened)

    def test_cgroup_mountinfo_path_traversal_is_rejected_before_metric_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t5 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("0::/job\n", encoding="utf-8")
            mount = root / "mount"
            outside_group = root / "outside" / "job"
            outside_group.mkdir(parents=True)
            (outside_group / "memory.current").write_text("999999\n", encoding="utf-8")
            (outside_group / "memory.peak").write_text("999999\n", encoding="utf-8")
            (outside_group / "memory.max").write_text("999999\n", encoding="utf-8")
            mount_with_traversal = f"{mount}/../outside"
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {mount_with_traversal} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            opened: list[Path] = []

            def reader(path: Path) -> str:
                opened.append(path)
                return path.read_text(encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=reader,
                get_ru_maxrss_kib=lambda: 12,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.cgroup_memory_current_bytes)
        self.assertFalse(any(path.name == "memory.current" for path in opened))

    def test_missing_rss_field_is_reported_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory) / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            (self_dir / "status").write_text("Name:\tfixture\n", encoding="utf-8")
            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.process_rss_bytes)
        self.assertEqual(snapshot.unavailable["process_rss_bytes"], "ValueError")
        self.assertIn("process_rss_bytes:ValueError", snapshot.diagnostics)

    def test_negative_ru_maxrss_is_unavailable_without_failing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory) / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            (self_dir / "status").write_text("VmRSS:\t7 kB\n", encoding="utf-8")
            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: -1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.process_rss_bytes, 7 * 1024)
        self.assertIsNone(snapshot.process_ru_maxrss_kib)
        self.assertIn("process_ru_maxrss_kib", snapshot.unavailable)

    def test_negative_cgroup_numeric_is_unavailable_and_nonnegative_values_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True)
            self_dir.joinpath("status").write_text("VmRSS:\t8 kB\n", encoding="utf-8")
            self_dir.joinpath("cgroup").write_text("0::/job\n", encoding="utf-8")
            mount = root / "cgroup-mount"
            group = mount / "job"
            group.mkdir(parents=True)
            self_dir.joinpath("mountinfo").write_text(
                f"29 23 0:26 / {mount} rw - cgroup2 cgroup rw\n",
                encoding="utf-8",
            )
            (group / "memory.current").write_text("-1\n", encoding="utf-8")
            (group / "memory.peak").write_text("0\n", encoding="utf-8")
            (group / "memory.max").write_text("42\n", encoding="utf-8")
            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 1,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.cgroup_memory_current_bytes)
        self.assertIn("cgroup_memory_current_bytes", snapshot.unavailable)
        self.assertEqual(snapshot.cgroup_memory_peak_bytes, 0)
        self.assertEqual(snapshot.cgroup_memory_limit_bytes, 42)
        self.assertFalse(snapshot.cgroup_memory_limit_unlimited)

    def test_resource_permission_errors_are_fail_soft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory) / "proc"

            def permission_denied(path: Path) -> str:
                raise PermissionError(str(path))

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=permission_denied,
                get_ru_maxrss_kib=lambda: 4,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.process_rss_bytes)
        self.assertEqual(snapshot.unavailable["process_rss_bytes"], "PermissionError")
        self.assertIsNone(snapshot.cgroup_version)
        self.assertIn("PermissionError", snapshot.unavailable["cgroup_version"])
        self.assertIn("PermissionError", snapshot.unavailable["cgroup_memory_current_bytes"])

    def test_non_linux_resource_snapshot_does_not_probe_proc_or_ru_maxrss(self) -> None:
        calls: list[Path] = []

        def unexpected_reader(path: Path) -> str:
            calls.append(path)
            raise AssertionError("non-Linux snapshot touched Linux procfs")

        snapshot = collect_resource_snapshot(
            captured_at=UTC,
            read_text=unexpected_reader,
            get_ru_maxrss_kib=lambda: (_ for _ in ()).throw(
                AssertionError("non-Linux snapshot queried ru_maxrss")
            ),
            system="Windows",
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(calls, [])
        self.assertIsNone(snapshot.process_rss_bytes)
        self.assertIsNone(snapshot.process_ru_maxrss_kib)
        self.assertIn("process_rss_bytes", snapshot.not_applicable)
        self.assertIn("cgroup_version", snapshot.not_applicable)

    def test_rss_and_ru_maxrss_have_separate_names_units_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            self_path = proc_root / "self"
            self_path.mkdir(parents=True)
            (self_path / "status").write_text("Name:\tfixture\nVmRSS:\t123 kB\n", encoding="utf-8")

            snapshot = collect_resource_snapshot(
                captured_at=UTC,
                proc_root=proc_root,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                get_ru_maxrss_kib=lambda: 456,
                system="Linux",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        encoded = snapshot.to_dict()
        self.assertEqual(encoded["process_rss_bytes"], 123 * 1024)
        self.assertEqual(encoded["process_ru_maxrss_kib"], 456)
        self.assertEqual(encoded["sources"]["process_rss_bytes"], "/proc/self/status:VmRSS")
        self.assertEqual(encoded["sources"]["process_ru_maxrss_kib"], "resource.getrusage(RUSAGE_SELF).ru_maxrss")
        self.assertIsNone(encoded["cgroup_version"])
        self.assertIn("cgroup_memory_current_bytes", encoded["unavailable"])


if __name__ == "__main__":
    unittest.main()
