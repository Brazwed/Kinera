"""CPU-only integrity tests for the P3e OpenCL event profiler."""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from profile_opencl_device import (  # noqa: E402
    CL_PROFILING_COMMAND_END,
    CL_PROFILING_COMMAND_QUEUED,
    CL_PROFILING_COMMAND_START,
    CL_PROFILING_COMMAND_SUBMIT,
    CL_PROFILING_INFO_NOT_AVAILABLE,
    CL_QUEUE_PROFILING_ENABLE,
    CL_QUEUE_PROPERTIES,
    CL_SUCCESS,
    OpenCLApiError,
    ProfilingOpenCLApi,
    _profile_durations,
    _source_fingerprint,
    _stats,
)

RESULT_PATH = ROOT / "experiments/results/opencl-rx550-p3e-device-profiling.json"
REPORT_PATH = ROOT / "docs/opencl-rx550-p3e-device-profiling.md"


class OpenCLEventProfilingTests(unittest.TestCase):
    def _api(self) -> tuple[Any, list[int]]:
        api = object.__new__(ProfilingOpenCLApi)
        api._capture = {
            "events": [], "host_components_ns": defaultdict(list)
        }
        api._stage = "forward"
        released: list[int] = []
        timestamps = {
            CL_PROFILING_COMMAND_QUEUED: 100,
            CL_PROFILING_COMMAND_SUBMIT: 110,
            CL_PROFILING_COMMAND_START: 130,
            CL_PROFILING_COMMAND_END: 170,
        }

        def get_profile(event, key, size, value, size_ret):
            ctypes.cast(value, ctypes.POINTER(ctypes.c_ulonglong))[0] = timestamps[key]
            return CL_SUCCESS

        def release_event(event):
            released.append(int(event.value or event))
            return CL_SUCCESS

        api._cl_get_event_profiling_info = get_profile
        api._cl_release_event = release_event
        return api, released

    def test_event_timestamps_are_derived_and_event_released_once(self) -> None:
        api, released = self._api()
        event = ctypes.c_void_p(41)

        api._profile_and_release_event(event, "kernel", {"host_enqueue_ns": 5})

        record = api._capture["events"][0]
        self.assertEqual(record["durations_ns"]["queued_to_submit_ns"], 10)
        self.assertEqual(record["durations_ns"]["submit_to_start_ns"], 20)
        self.assertEqual(record["durations_ns"]["kernel_execution_ns"], 40)
        self.assertTrue(record["profiling_available"])
        self.assertTrue(record["event_release_succeeded"])
        self.assertEqual(released, [41])
        self.assertIsNone(event.value)

    def test_unavailable_timestamp_is_explicit_and_still_releases_event(self) -> None:
        api, released = self._api()
        original = api._cl_get_event_profiling_info

        def unavailable_submit(event, key, size, value, size_ret):
            if key == CL_PROFILING_COMMAND_SUBMIT:
                return CL_PROFILING_INFO_NOT_AVAILABLE
            return original(event, key, size, value, size_ret)

        api._cl_get_event_profiling_info = unavailable_submit
        event = ctypes.c_void_p(42)

        api._profile_and_release_event(event, "kernel", {})

        record = api._capture["events"][0]
        self.assertFalse(record["profiling_available"])
        self.assertTrue(record["profiling_info_not_available"])
        self.assertEqual(
            record["profile_query_status"],
            [{"field": "submit_ns", "status": CL_PROFILING_INFO_NOT_AVAILABLE}],
        )
        self.assertEqual(released, [42])

    def test_event_release_failure_is_reported_after_one_release_attempt(self) -> None:
        api, released = self._api()

        def fail_release(event):
            released.append(int(event.value))
            return -5

        api._cl_release_event = fail_release
        event = ctypes.c_void_p(43)
        with self.assertRaises(OpenCLApiError):
            api._profile_and_release_event(event, "kernel", {})

        self.assertEqual(released, [43])
        self.assertIsNone(event.value)
        self.assertFalse(api._capture["events"][0]["event_release_succeeded"])

    def test_finish_failure_releases_enqueued_event_once(self) -> None:
        api, released = self._api()

        def enqueue(*args):
            ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_void_p))[0] = 44
            return CL_SUCCESS

        api._cl_enqueue_nd_range_kernel = enqueue
        api._cl_finish = lambda queue: -5
        api._cl_release_event = lambda event: (released.append(int(event.value)), CL_SUCCESS)[1]

        with self.assertRaisesRegex(OpenCLApiError, "clFinish"):
            ProfilingOpenCLApi.enqueue_1d(api, 1, 2, 1)

        self.assertEqual(released, [44])

    def test_enqueue_failure_propagates_without_query_or_release(self) -> None:
        api, released = self._api()
        finish_calls: list[int] = []
        api._cl_enqueue_nd_range_kernel = lambda *args: -5
        api._cl_finish = lambda queue: finish_calls.append(queue) or CL_SUCCESS
        api._cl_release_event = lambda event: released.append(int(event.value)) or CL_SUCCESS

        with self.assertRaisesRegex(OpenCLApiError, "clEnqueueNDRangeKernel"):
            ProfilingOpenCLApi.enqueue_1d(api, 1, 2, 1)

        self.assertEqual(finish_calls, [])
        self.assertEqual(released, [])

    def test_profiling_query_exception_still_releases_event_and_propagates(self) -> None:
        api, released = self._api()

        def fail_query(*args):
            raise RuntimeError("profiling query raised")

        api._cl_get_event_profiling_info = fail_query
        event = ctypes.c_void_p(46)
        with self.assertRaisesRegex(RuntimeError, "profiling query raised"):
            api._profile_and_release_event(event, "kernel", {})

        self.assertEqual(released, [46])
        self.assertIsNone(event.value)
        self.assertTrue(api._capture["events"][0]["event_release_succeeded"])

    def test_profile_queue_is_enabled_and_verified(self) -> None:
        api = object.__new__(ProfilingOpenCLApi)
        api.queue_properties = None
        queue_flags: list[int] = []

        def create_queue(context, device, properties, status):
            queue_flags.append(int(properties))
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = CL_SUCCESS
            return 45

        def query_queue(queue, key, size, value, size_ret):
            self.assertEqual(key, CL_QUEUE_PROPERTIES)
            ctypes.cast(value, ctypes.POINTER(ctypes.c_ulonglong))[0] = (
                CL_QUEUE_PROFILING_ENABLE
            )
            return CL_SUCCESS

        api._cl_create_command_queue = create_queue
        api._cl_get_command_queue_info = query_queue
        api.release_command_queue = lambda queue: None

        self.assertEqual(ProfilingOpenCLApi.create_command_queue(api, 1, 2), 45)
        self.assertEqual(queue_flags, [CL_QUEUE_PROFILING_ENABLE])
        self.assertEqual(api.queue_properties, CL_QUEUE_PROFILING_ENABLE)

    def test_invalid_timestamp_order_is_not_reported_as_duration(self) -> None:
        self.assertIsNone(
            _profile_durations(
                {"queued_ns": 20, "submit_ns": 10, "start_ns": 30, "end_ns": 40}
            )
        )

    def test_statistics_are_deterministic(self) -> None:
        result = _stats([1, 2, 3, 4, 5])
        self.assertEqual(result["median"], 3)
        self.assertEqual(result["q1"], 2)
        self.assertEqual(result["q3"], 4)
        self.assertEqual(result["iqr"], 2)


class P3eResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_result_records_requested_base_processes_and_shapes(self) -> None:
        self.assertEqual(
            self.result["code"]["base_commit"],
            "1759bd196dee268e4a9e0e80f6820e0a37db36ed",
        )
        self.assertEqual(self.result["protocol"]["process_runs"], 3)
        self.assertEqual(len({worker["pid"] for worker in self.result["workers"]}), 3)
        self.assertEqual(
            [(case["label"], case["weights_shape"]) for case in self.result["workers"][0]["cases"]],
            [
                ("1x1", [1, 1]),
                ("51x26", [51, 26]),
                ("3932x26", [3932, 26]),
                ("26x3932", [26, 3932]),
            ],
        )
        self.assertEqual(
            self.result["workers"][0]["environment"]["device"]["pci_bdf"],
            "0000:03:00.0",
        )
        self.assertEqual(self.result["protocol"]["warmups_per_operation"], 5)
        self.assertEqual(self.result["protocol"]["timed_repetitions_per_operation"], 30)
        self.assertEqual(self.result["protocol"]["dtype"], "float64")
        self.assertEqual(self.result["protocol"]["atol"], 1e-12)
        self.assertEqual(self.result["protocol"]["rtol"], 1e-12)

    def test_total_event_counts_and_numerical_errors(self) -> None:
        self.assertEqual(len(self.result["workers"]), 3)
        self.assertEqual(len({worker["pid"] for worker in self.result["workers"]}), 3)
        kernel_events = 0
        transfer_events = 0
        numerical_errors = []
        for worker in self.result["workers"]:
            for case in worker["cases"]:
                for operation in case["operations"].values():
                    profile = operation["profiled"]
                    for per_sample in profile["numerical_max_abs_error"].values():
                        numerical_errors.extend(per_sample.values())
                    for sample in profile["samples"]:
                        kernel_events += sum(event["kind"] == "kernel" for event in sample["events"])
                for transfer in case["profiled_blocking_transfers"].values():
                    if not isinstance(transfer, dict):
                        continue
                    for sample in transfer.get("samples", []):
                        transfer_events += len(sample["events"])
        self.assertEqual(kernel_events, 2160)
        self.assertEqual(transfer_events, 720)
        self.assertTrue(numerical_errors)
        self.assertEqual(max(numerical_errors), 0.0)

    def test_artifact_hashes_match_report_and_current_sources(self) -> None:
        report = REPORT_PATH.read_text(encoding="utf-8")
        result_hash = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertIn(result_hash, report)
        self.assertEqual(
            self.result["code"]["source_fingerprint_sha256"], _source_fingerprint()
        )

    def test_samples_release_events_and_preserve_disjoint_host_accounting(self) -> None:
        repetitions = self.result["protocol"]["timed_repetitions_per_operation"]
        tolerance = self.result["protocol"]["atol"]
        for worker in self.result["workers"]:
            for case in worker["cases"]:
                self.assertEqual(case["profiling_queue_properties"] & CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PROFILING_ENABLE)
                for operation, operation_data in case["operations"].items():
                    profile = operation_data["profiled"]
                    self.assertEqual(len(profile["samples"]), repetitions)
                    for sample in profile["samples"]:
                        self.assertEqual(
                            sample["host_operation_ns"],
                            sample["host_api_component_sum_ns"]
                            + sample["linear_array_validation_ns"]
                            + sample["unattributed_python_and_layer_ns"],
                        )
                        self.assertEqual(
                            len(sample["events"]), 3 if operation == "resident_cycle" else 1
                        )
                        for event in sample["events"]:
                            self.assertTrue(event["event_release_succeeded"])
                            self.assertTrue(event["profiling_available"])
                            calculated = _profile_durations(event["timestamps_ns"])
                            self.assertIsNotNone(calculated)
                            if event["kind"] == "kernel":
                                self.assertEqual(event["durations_ns"], calculated)
                            else:
                                self.assertEqual(
                                    event["durations_ns"]["command_execution_ns"],
                                    calculated["kernel_execution_ns"],
                                )
                    for by_sample in profile["numerical_max_abs_error"].values():
                        for error in by_sample.values():
                            self.assertLessEqual(error, tolerance)
                transfers = case["profiled_blocking_transfers"]
                for kind in ("h2d", "d2h"):
                    self.assertEqual(len(transfers[kind]["samples"]), repetitions)
                    for sample in transfers[kind]["samples"]:
                        self.assertEqual(len(sample["events"]), 1)
                        event = sample["events"][0]
                        self.assertTrue(event["event_release_succeeded"])
                        self.assertTrue(event["profiling_available"])
                self.assertLessEqual(transfers["d2h_roundtrip_max_abs_error"], tolerance)

    def test_aggregate_host_statistics_recompute_from_raw_samples(self) -> None:
        for label, aggregate_case in self.result["aggregate"].items():
            worker_cases = [
                next(case for case in worker["cases"] if case["label"] == label)
                for worker in self.result["workers"]
            ]
            for operation, aggregate_operation in aggregate_case["operations"].items():
                for condition in ("profiled", "control"):
                    raw = [
                        sample["host_operation_ns"]
                        for case in worker_cases
                        for sample in case["operations"][operation][condition]["samples"]
                    ]
                    self.assertEqual(
                        aggregate_operation["conditions"][condition]["host_operation_ns"],
                        _stats(raw),
                    )

    def test_aggregate_device_and_transfer_statistics_recompute_from_raw_samples(self) -> None:
        for label, aggregate_case in self.result["aggregate"].items():
            worker_cases = [
                next(case for case in worker["cases"] if case["label"] == label)
                for worker in self.result["workers"]
            ]
            for operation, aggregate_operation in aggregate_case["operations"].items():
                stages = ("forward", "backward", "step") if operation == "resident_cycle" else (operation,)
                for stage in stages:
                    events = [
                        event
                        for case in worker_cases
                        for sample in case["operations"][operation]["profiled"]["samples"]
                        for event in sample["events"]
                        if event["stage"] == stage
                    ]
                    for field, duration_name in (
                        ("kernel_execution_ns", "kernel_execution_ns"),
                        ("queued_to_submit_ns", "queued_to_submit_ns"),
                        ("submit_to_start_ns", "submit_to_start_ns"),
                    ):
                        values = [event["durations_ns"][duration_name] for event in events]
                        self.assertEqual(
                            aggregate_operation["device_events"][stage][field], _stats(values)
                        )

                paired_deltas = [
                    profiled["host_operation_ns"] - control["host_operation_ns"]
                    for case in worker_cases
                    for profiled, control in zip(
                        case["operations"][operation]["profiled"]["samples"],
                        case["operations"][operation]["control"]["samples"],
                    )
                ]
                self.assertEqual(
                    aggregate_operation["paired_profiled_minus_control_host_ns"],
                    _stats(paired_deltas),
                )

            for kind in ("h2d", "d2h"):
                samples = [
                    sample
                    for case in worker_cases
                    for sample in case["profiled_blocking_transfers"][kind]["samples"]
                ]
                device_durations = [
                    event["durations_ns"]["command_execution_ns"]
                    for sample in samples
                    for event in sample["events"]
                ]
                aggregate_transfer = aggregate_case["profiled_blocking_transfers"][kind]
                self.assertEqual(
                    aggregate_transfer["host_blocking_call_ns"],
                    _stats([sample["host_blocking_call_ns"] for sample in samples]),
                )
                self.assertEqual(
                    aggregate_transfer["device_event_execution_ns"], _stats(device_durations)
                )


if __name__ == "__main__":
    unittest.main()
