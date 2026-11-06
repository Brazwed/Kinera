"""Profile OpenCL command events on the Kinera RX 550 backend.

This is an experiment-only adapter. It enables queue profiling and requests
events without changing the production OpenCL API, backend, or kernels.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from benchmark_opencl_linear import (  # noqa: E402
    ATOL,
    CASES,
    LEARNING_RATE,
    RTOL,
    _case_data,
    _device_record,
    _flatten,
    _max_abs_error,
    _system_snapshot,
)
from kinera.backends.cpu.backend import CPUBackend  # noqa: E402
from kinera.backends.opencl.api import (  # noqa: E402
    CL_SUCCESS,
    CL_TRUE,
    OpenCLApi,
    OpenCLApiError,
)
from kinera.backends.opencl.backend import OpenCLBackend  # noqa: E402
from kinera.backends.opencl.runtime import OpenCLRuntime  # noqa: E402
from kinera.nn.layers.linear import Linear  # noqa: E402


CL_QUEUE_PROFILING_ENABLE = 1 << 1
CL_QUEUE_PROPERTIES = 0x1093
CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_PROFILING_INFO_NOT_AVAILABLE = -7

PROFILE_KEYS = (
    ("queued_ns", CL_PROFILING_COMMAND_QUEUED),
    ("submit_ns", CL_PROFILING_COMMAND_SUBMIT),
    ("start_ns", CL_PROFILING_COMMAND_START),
    ("end_ns", CL_PROFILING_COMMAND_END),
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile for an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {"samples": 0, "median": 0, "q1": 0, "q3": 0, "iqr": 0}
    numbers = [float(value) for value in values]
    q1 = _percentile(numbers, 0.25)
    q3 = _percentile(numbers, 0.75)
    return {
        "samples": len(numbers),
        "median": statistics.median(numbers),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(numbers),
        "max": max(numbers),
    }


def _profile_durations(timestamps: dict[str, int]) -> dict[str, int] | None:
    required = ("queued_ns", "submit_ns", "start_ns", "end_ns")
    if any(key not in timestamps for key in required):
        return None
    queued, submitted, started, ended = (timestamps[key] for key in required)
    if not queued <= submitted <= started <= ended:
        return None
    return {
        "queued_to_submit_ns": submitted - queued,
        "submit_to_start_ns": started - submitted,
        "queued_to_start_ns": started - queued,
        "kernel_execution_ns": ended - started,
        "queued_to_end_ns": ended - queued,
    }


class LayerValidationRecorder:
    """Collect Linear's Python-side input/output validation spans."""

    def __init__(self) -> None:
        self.active = False
        self.stage = "operation"
        self._values: dict[str, list[int]] = defaultdict(list)

    def begin(self, stage: str = "operation") -> None:
        self.active = True
        self.stage = stage
        self._values = defaultdict(list)

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def record(self, duration_ns: int) -> None:
        if self.active:
            self._values[self.stage].append(duration_ns)

    def end(self) -> dict[str, list[int]]:
        self.active = False
        return {stage: list(values) for stage, values in self._values.items()}


class ObservedLinear(Linear):
    """Linear subclass that times validation without changing its result."""

    def __init__(self, *args: Any, validation_recorder: LayerValidationRecorder, **kwargs: Any) -> None:
        self._validation_recorder = validation_recorder
        super().__init__(*args, **kwargs)

    def _validate_backend_array(
        self, value: object, shape: tuple[int, ...], name: str
    ) -> Any:
        started = time.perf_counter_ns()
        try:
            return super()._validate_backend_array(value, shape, name)
        finally:
            self._validation_recorder.record(time.perf_counter_ns() - started)


class ProfilingOpenCLApi(OpenCLApi):
    """OpenCL C API adapter which attaches and releases profiling events."""

    def __init__(self) -> None:
        super().__init__()
        void_p = ctypes.c_void_p
        uint = ctypes.c_uint
        int_ = ctypes.c_int
        size_t = ctypes.c_size_t
        self._cl_get_command_queue_info = self._bind(
            "clGetCommandQueueInfo",
            int_,
            [void_p, uint, size_t, void_p, ctypes.POINTER(size_t)],
        )
        self._cl_get_event_profiling_info = self._bind(
            "clGetEventProfilingInfo",
            int_,
            [void_p, uint, size_t, void_p, ctypes.POINTER(size_t)],
        )
        self._cl_release_event = self._bind("clReleaseEvent", int_, [void_p])
        self.queue_properties: int | None = None
        self._capture: dict[str, Any] | None = None
        self._stage = "operation"

    def create_command_queue(self, context: int, device: int) -> int:
        status = ctypes.c_int()
        queue = self._cl_create_command_queue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE,
            ctypes.byref(status),
        )
        self._check(status.value, "clCreateCommandQueue(CL_QUEUE_PROFILING_ENABLE)")
        if not queue:
            raise OpenCLApiError("profiling clCreateCommandQueue returned a null queue")
        queue_handle = int(queue)
        properties = ctypes.c_ulonglong()
        try:
            self._check(
                self._cl_get_command_queue_info(
                    queue_handle,
                    CL_QUEUE_PROPERTIES,
                    ctypes.sizeof(properties),
                    ctypes.byref(properties),
                    None,
                ),
                "clGetCommandQueueInfo(CL_QUEUE_PROPERTIES)",
            )
            self.queue_properties = int(properties.value)
            if not self.queue_properties & CL_QUEUE_PROFILING_ENABLE:
                raise OpenCLApiError(
                    "OpenCL accepted the profiling queue but did not report "
                    "CL_QUEUE_PROFILING_ENABLE"
                )
        except BaseException as error:
            try:
                self.release_command_queue(queue_handle)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        return queue_handle

    def begin_capture(self, stage: str = "operation") -> None:
        if self._capture is not None:
            raise RuntimeError("OpenCL event capture is already active")
        self._stage = stage
        self._capture = {"events": [], "host_components_ns": defaultdict(list)}

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def end_capture(self) -> dict[str, Any]:
        if self._capture is None:
            raise RuntimeError("OpenCL event capture is not active")
        capture = self._capture
        self._capture = None
        return {
            "events": list(capture["events"]),
            "host_components_ns": {
                name: list(values)
                for name, values in capture["host_components_ns"].items()
            },
        }

    def _record_component(self, name: str, duration_ns: int) -> None:
        if self._capture is not None:
            self._capture["host_components_ns"][name].append(duration_ns)

    def _record_event(self, kind: str, event: dict[str, Any]) -> None:
        if self._capture is not None:
            durations = event.get("durations_ns")
            if kind != "kernel" and durations is not None:
                durations = dict(durations)
                durations["command_execution_ns"] = durations.pop("kernel_execution_ns")
                event["durations_ns"] = durations
            self._capture["events"].append(
                {"stage": self._stage, "kind": kind, **event}
            )

    def _timed_call(self, name: str, function: Callable[..., Any], *args: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return function(*args)
        finally:
            self._record_component(name, time.perf_counter_ns() - started)

    def create_buffer(self, context: int, nbytes: int) -> int:
        return self._timed_call("create_buffer_api_ns", super().create_buffer, context, nbytes)

    def build_program(self, context: int, device: int, source: str, options: str) -> int:
        return self._timed_call(
            "build_program_api_ns", super().build_program, context, device, source, options
        )

    def create_kernel(self, program: int, name: str) -> int:
        return self._timed_call("create_kernel_api_ns", super().create_kernel, program, name)

    def set_kernel_buffer(self, kernel: int, index: int, buffer: int) -> None:
        self._timed_call("kernel_argument_api_ns", super().set_kernel_buffer, kernel, index, buffer)

    def set_kernel_int(self, kernel: int, index: int, value: int) -> None:
        self._timed_call("kernel_argument_api_ns", super().set_kernel_int, kernel, index, value)

    def set_kernel_double(self, kernel: int, index: int, value: float) -> None:
        self._timed_call("kernel_argument_api_ns", super().set_kernel_double, kernel, index, value)

    def release_kernel(self, handle: int) -> None:
        self._timed_call("release_kernel_api_ns", super().release_kernel, handle)

    def release_mem(self, handle: int) -> None:
        self._timed_call("release_mem_api_ns", super().release_mem, handle)

    def _query_event(self, event: ctypes.c_void_p) -> dict[str, Any]:
        values: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for name, key in PROFILE_KEYS:
            value = ctypes.c_ulonglong()
            status = int(
                self._cl_get_event_profiling_info(
                    event,
                    key,
                    ctypes.sizeof(value),
                    ctypes.byref(value),
                    None,
                )
            )
            statuses[name] = status
            if status == CL_SUCCESS:
                values[name] = int(value.value)
        if len(values) != len(PROFILE_KEYS):
            unavailable = [
                {"field": name, "status": status}
                for name, status in statuses.items()
                if status != CL_SUCCESS
            ]
            return {
                "profiling_available": False,
                "timestamps_ns": values,
                "profile_query_status": unavailable,
                "profiling_info_not_available": any(
                    item["status"] == CL_PROFILING_INFO_NOT_AVAILABLE
                    for item in unavailable
                ),
            }
        durations = _profile_durations(values)
        return {
            "profiling_available": durations is not None,
            "timestamps_ns": values,
            "durations_ns": durations,
            "profile_order_valid": durations is not None,
        }

    def _release_event(self, event: ctypes.c_void_p) -> None:
        self._check(self._cl_release_event(event), "clReleaseEvent")

    def _profile_and_release_event(
        self, event: ctypes.c_void_p, kind: str, host_timings: dict[str, int]
    ) -> tuple[int, int]:
        query_started = time.perf_counter_ns()
        query_error: BaseException | None = None
        event_record: dict[str, Any] = {}
        try:
            event_record = self._query_event(event)
        except BaseException as error:
            query_error = error
        query_ns = time.perf_counter_ns() - query_started
        release_started = time.perf_counter_ns()
        release_error: BaseException | None = None
        try:
            self._release_event(event)
        except BaseException as error:
            release_error = error
        finally:
            # The release attempt happened even if the driver reported failure;
            # never retry it implicitly from the caller's exception path.
            event.value = None
        release_ns = time.perf_counter_ns() - release_started
        event_record.update(host_timings)
        event_record["event_release_succeeded"] = release_error is None
        self._record_event(kind, event_record)
        self._record_component("event_profile_query_ns", query_ns)
        self._record_component("event_release_ns", release_ns)
        if query_error is not None and release_error is not None:
            raise query_error from release_error
        if query_error is not None:
            raise query_error
        if release_error is not None:
            raise release_error
        return query_ns, release_ns

    def enqueue_1d(self, queue: int, kernel: int, global_size: int) -> None:
        if self._capture is None:
            return super().enqueue_1d(queue, kernel, global_size)
        total_started = time.perf_counter_ns()
        event = ctypes.c_void_p()
        enqueue_ns = 0
        finish_ns = 0
        profile_query_ns = 0
        event_release_ns = 0
        try:
            work_size = ctypes.c_size_t(global_size)
            enqueue_started = time.perf_counter_ns()
            status = self._cl_enqueue_nd_range_kernel(
                queue,
                kernel,
                1,
                None,
                ctypes.byref(work_size),
                None,
                0,
                None,
                ctypes.byref(event),
            )
            enqueue_ns = time.perf_counter_ns() - enqueue_started
            self._check(status, "clEnqueueNDRangeKernel")
            if not event.value:
                raise OpenCLApiError("clEnqueueNDRangeKernel returned no profiling event")

            finish_started = time.perf_counter_ns()
            finish_status = self._cl_finish(queue)
            finish_ns = time.perf_counter_ns() - finish_started
            self._check(finish_status, "clFinish")

            profile_query_ns, event_release_ns = self._profile_and_release_event(
                event,
                "kernel",
                {"host_enqueue_ns": enqueue_ns, "host_cl_finish_ns": finish_ns},
            )
            event = ctypes.c_void_p()
        except BaseException as error:
            if event.value:
                try:
                    self._release_event(event)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise
        finally:
            total_ns = time.perf_counter_ns() - total_started
            accounted = enqueue_ns + finish_ns + profile_query_ns + event_release_ns
            if accounted > total_ns:
                raise RuntimeError("nested OpenCL enqueue timing components overlap")
            self._record_component("kernel_enqueue_native_ns", enqueue_ns)
            self._record_component("cl_finish_host_wait_ns", finish_ns)
            self._record_component("enqueue_other_host_ns", total_ns - accounted)

    def _blocking_transfer(
        self,
        kind: str,
        queue: int,
        buffer: int,
        host_address: int,
        nbytes: int,
    ) -> None:
        if self._capture is None:
            if kind == "h2d":
                return super().write_buffer(queue, buffer, host_address, nbytes)
            return super().read_buffer(queue, buffer, host_address, nbytes)

        total_started = time.perf_counter_ns()
        event = ctypes.c_void_p()
        native_ns = 0
        if kind == "h2d":
            operation = "clEnqueueWriteBuffer"
            enqueue = self._cl_enqueue_write_buffer
        else:
            operation = "clEnqueueReadBuffer"
            enqueue = self._cl_enqueue_read_buffer
        try:
            native_started = time.perf_counter_ns()
            status = enqueue(
                queue,
                buffer,
                CL_TRUE,
                0,
                nbytes,
                host_address,
                0,
                None,
                ctypes.byref(event),
            )
            native_ns = time.perf_counter_ns() - native_started
            self._check(status, operation)
            if not event.value:
                raise OpenCLApiError(f"{operation} returned no profiling event")
            profile_query_ns, event_release_ns = self._profile_and_release_event(
                event,
                kind,
                {"host_blocking_api_ns": native_ns, "bytes": nbytes},
            )
            event = ctypes.c_void_p()
        except BaseException as error:
            if event.value:
                try:
                    self._release_event(event)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise
        finally:
            total_ns = time.perf_counter_ns() - total_started
            accounted = native_ns + profile_query_ns + event_release_ns
            if accounted > total_ns:
                raise RuntimeError("nested OpenCL transfer timing components overlap")
            self._record_component(f"{kind}_blocking_api_inclusive_ns", total_ns)
            self._record_component(f"{kind}_transfer_native_call_ns", native_ns)
            self._record_component(f"{kind}_transfer_other_host_ns", total_ns - accounted)

    def write_buffer(self, queue: int, buffer: int, host_address: int, nbytes: int) -> None:
        self._blocking_transfer("h2d", queue, buffer, host_address, nbytes)

    def read_buffer(self, queue: int, buffer: int, host_address: int, nbytes: int) -> None:
        self._blocking_transfer("d2h", queue, buffer, host_address, nbytes)


def _sample_record(
    elapsed_ns: int,
    validation: dict[str, list[int]],
    capture: dict[str, Any] | None,
) -> dict[str, Any]:
    validation_total = sum(sum(values) for values in validation.values())
    components = capture["host_components_ns"] if capture is not None else {}
    api_total = sum(sum(values) for values in components.values())
    if api_total + validation_total > elapsed_ns:
        raise RuntimeError("profile components exceed the inclusive Linear wall time")
    return {
        "host_operation_ns": elapsed_ns,
        "linear_array_validation_ns": validation_total,
        "host_api_components_ns": {
            name: values for name, values in components.items()
        },
        "host_api_component_sum_ns": api_total,
        "unattributed_python_and_layer_ns": elapsed_ns - api_total - validation_total,
        "validation_by_stage_ns": validation,
        "events": capture["events"] if capture is not None else [],
    }


def _start_measurement(
    api: OpenCLApi | ProfilingOpenCLApi,
    observer: LayerValidationRecorder,
    stage: str,
    *,
    capture_events: bool,
) -> None:
    observer.begin(stage)
    if capture_events:
        assert isinstance(api, ProfilingOpenCLApi)
        api.begin_capture(stage)


def _finish_measurement(
    api: OpenCLApi | ProfilingOpenCLApi,
    observer: LayerValidationRecorder,
    *,
    capture_events: bool,
) -> tuple[dict[str, list[int]], dict[str, Any] | None]:
    validation = observer.end()
    capture = api.end_capture() if capture_events else None
    return validation, capture


def _set_stage(
    api: OpenCLApi | ProfilingOpenCLApi,
    observer: LayerValidationRecorder,
    stage: str,
    *,
    capture_events: bool,
) -> None:
    observer.set_stage(stage)
    if capture_events:
        assert isinstance(api, ProfilingOpenCLApi)
        api.set_stage(stage)


def _close_if_open(value: Any) -> None:
    if value is not None and hasattr(value, "close") and not value.closed:
        value.close()


def _make_variant(
    case: dict[str, Any],
    *,
    profiled: bool,
) -> dict[str, Any]:
    runtime_started = time.perf_counter_ns()
    api: OpenCLApi | ProfilingOpenCLApi
    api = ProfilingOpenCLApi() if profiled else OpenCLApi()
    runtime = OpenCLRuntime(api=api)
    backend = OpenCLBackend(runtime=runtime)
    device = runtime.require_device(0, require_fp64=True)
    runtime_discovery_ns = time.perf_counter_ns() - runtime_started
    if device.pci_bdf != "0000:03:00.0" or "Radeon RX 550" not in device.name:
        backend.close()
        raise RuntimeError(
            f"P3e must run only on RX 550 at 0000:03:00.0; got "
            f"{device.name!r} ({device.pci_bdf!r})"
        )
    queue_started = time.perf_counter_ns()
    _ = backend.memory.queue
    queue_creation_ns = time.perf_counter_ns() - queue_started
    if profiled and (
        not isinstance(api, ProfilingOpenCLApi)
        or api.queue_properties is None
        or not api.queue_properties & CL_QUEUE_PROFILING_ENABLE
    ):
        backend.close()
        raise RuntimeError("profiling queue properties were not confirmed")

    observer = LayerValidationRecorder()
    started = time.perf_counter_ns()
    try:
        layer = ObservedLinear(
            case["input_size"],
            case["output_size"],
            seed=case["seed"],
            backend=backend,
            validation_recorder=observer,
        )
        inputs = backend.buffer_from_host(
            case["inputs"], shape=(case["input_size"],), dtype="float64"
        )
        grad_output = backend.buffer_from_host(
            case["grad_output"], shape=(case["output_size"],), dtype="float64"
        )
    except BaseException as error:
        try:
            backend.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    initialization_ns = time.perf_counter_ns() - started
    return {
        "profiled": profiled,
        "api": api,
        "runtime": runtime,
        "backend": backend,
        "device": device,
        "observer": observer,
        "layer": layer,
        "inputs": inputs,
        "grad_output": grad_output,
        "runtime_discovery_ns": runtime_discovery_ns,
        "queue_creation_ns": queue_creation_ns,
        "layer_and_input_setup_ns": initialization_ns,
    }


def _compile_programs(variant: dict[str, Any]) -> dict[str, int]:
    api = variant["api"]
    backend = variant["backend"]
    samples: dict[str, int] = {}
    for operation, filename in (
        ("forward", "linear_forward.cl"),
        ("backward", "linear_backward.cl"),
        ("step", "linear_step.cl"),
    ):
        if variant["profiled"]:
            assert isinstance(api, ProfilingOpenCLApi)
            api.begin_capture(f"compile_{operation}")
        started = time.perf_counter_ns()
        backend._get_program(filename)
        samples[operation] = time.perf_counter_ns() - started
        if variant["profiled"]:
            api.end_capture()
    return samples


def _measure_one(
    variant: dict[str, Any],
    stage: str,
    callback: Callable[[], Any],
    *,
    capture_events: bool,
) -> tuple[Any, dict[str, Any]]:
    api = variant["api"]
    observer = variant["observer"]
    _start_measurement(api, observer, stage, capture_events=capture_events)
    started = time.perf_counter_ns()
    try:
        result = callback()
    except BaseException:
        if capture_events:
            api.end_capture()
        observer.end()
        raise
    elapsed_ns = time.perf_counter_ns() - started
    validation, capture = _finish_measurement(
        api, observer, capture_events=capture_events
    )
    return result, _sample_record(elapsed_ns, validation, capture)


def _expected_cycle(
    case: dict[str, Any], cycles: int, observed_indices: set[int]
) -> list[dict[str, Any]]:
    cpu = CPUBackend()
    layer = Linear(
        case["input_size"], case["output_size"], seed=case["seed"], backend=cpu
    )
    snapshots = []
    try:
        for index in range(cycles):
            output = layer.forward(case["inputs"])
            grad_inputs = layer.backward(case["grad_output"])
            snapshot = {
                "output": deepcopy(output),
                "grad_weights": deepcopy(layer.grad_weights),
                "grad_bias": deepcopy(layer.grad_bias),
                "grad_inputs": deepcopy(grad_inputs),
            }
            layer.step(LEARNING_RATE)
            snapshot["weights"] = deepcopy(layer.weights)
            snapshot["bias"] = deepcopy(layer.bias)
            if index in observed_indices:
                snapshots.append({"cycle_index": index, **snapshot})
        return snapshots
    finally:
        layer.close()


def _cpu_single_step(case: dict[str, Any]) -> dict[str, Any]:
    cpu = CPUBackend()
    layer = Linear(
        case["input_size"], case["output_size"], seed=case["seed"], backend=cpu
    )
    try:
        output = deepcopy(layer.forward(case["inputs"]))
        grad_inputs = deepcopy(layer.backward(case["grad_output"]))
        expected = {
            "output": output,
            "grad_weights": deepcopy(layer.grad_weights),
            "grad_bias": deepcopy(layer.grad_bias),
            "grad_inputs": grad_inputs,
        }
        layer.step(LEARNING_RATE)
        expected["weights"] = deepcopy(layer.weights)
        expected["bias"] = deepcopy(layer.bias)
        return expected
    finally:
        layer.close()


def _prepare_backward(variant: dict[str, Any]) -> None:
    layer = variant["layer"]
    _close_if_open(layer.output)
    layer.forward(variant["inputs"])


def _reset_parameters(variant: dict[str, Any], case: dict[str, Any]) -> None:
    api = variant["api"]
    layer = variant["layer"]
    flat_weights = _flatten(case["weights"])
    weight_array = (ctypes.c_double * len(flat_weights))(*flat_weights)
    bias_array = (ctypes.c_double * len(case["bias"]))(*case["bias"])
    api.write_buffer(
        variant["backend"].memory.queue,
        layer.weights.handle,
        ctypes.addressof(weight_array),
        layer.weights.nbytes,
    )
    api.write_buffer(
        variant["backend"].memory.queue,
        layer.bias.handle,
        ctypes.addressof(bias_array),
        layer.bias.nbytes,
    )


def _measure_cycle_once(
    variant: dict[str, Any],
    cycle_index: int,
    expected: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    layer = variant["layer"]
    _close_if_open(layer.output)
    _start_measurement(
        variant["api"], variant["observer"], "cycle",
        capture_events=variant["profiled"],
    )
    total_started = time.perf_counter_ns()
    stage_elapsed: dict[str, int] = {}
    output = None
    grad_inputs = None
    try:
        _set_stage(
            variant["api"], variant["observer"], "forward",
            capture_events=variant["profiled"],
        )
        phase_started = time.perf_counter_ns()
        output = layer.forward(variant["inputs"])
        stage_elapsed["forward"] = time.perf_counter_ns() - phase_started

        _set_stage(
            variant["api"], variant["observer"], "backward",
            capture_events=variant["profiled"],
        )
        phase_started = time.perf_counter_ns()
        grad_inputs = layer.backward(variant["grad_output"])
        stage_elapsed["backward"] = time.perf_counter_ns() - phase_started

        _set_stage(
            variant["api"], variant["observer"], "step",
            capture_events=variant["profiled"],
        )
        phase_started = time.perf_counter_ns()
        layer.step(LEARNING_RATE)
        stage_elapsed["step"] = time.perf_counter_ns() - phase_started
    except BaseException:
        if variant["profiled"]:
            variant["api"].end_capture()
        variant["observer"].end()
        _close_if_open(grad_inputs)
        _close_if_open(output)
        raise

    elapsed_ns = time.perf_counter_ns() - total_started
    validation, capture = _finish_measurement(
        variant["api"], variant["observer"],
        capture_events=variant["profiled"],
    )
    record = _sample_record(elapsed_ns, validation, capture)
    record["public_linear_stage_ns"] = stage_elapsed

    errors: dict[str, float] = {}
    if expected is not None:
        actual = {
            "output": variant["backend"].buffer_to_host(output),
            "grad_weights": variant["backend"].buffer_to_host(layer.grad_weights),
            "grad_bias": variant["backend"].buffer_to_host(layer.grad_bias),
            "grad_inputs": variant["backend"].buffer_to_host(grad_inputs),
            "weights": variant["backend"].buffer_to_host(layer.weights),
            "bias": variant["backend"].buffer_to_host(layer.bias),
        }
        errors = {
            name: _max_abs_error(actual[name], expected[name], f"cycle {cycle_index} {name}")
            for name in actual
        }

    grad_inputs.close()
    output.close()
    return record, errors


def _measure_operation_once(
    variant: dict[str, Any],
    operation: str,
    case: dict[str, Any],
    expected: dict[str, Any],
    cycle_expected: dict[str, Any] | None,
    index: int,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    layer = variant["layer"]
    capture_events = variant["profiled"]
    verify = index >= warmups and index in {warmups, warmups + repetitions - 1}

    if operation == "resident_cycle":
        return _measure_cycle_once(variant, index, cycle_expected if verify else None)

    if operation == "forward":
        _close_if_open(layer.output)
        output, record = _measure_one(
            variant, "forward",
            lambda: layer.forward(variant["inputs"]),
            capture_events=capture_events,
        )
        errors = {}
        if verify:
            errors["output"] = _max_abs_error(
                variant["backend"].buffer_to_host(output),
                expected["output"],
                "forward output",
            )
        output.close()
        return record, errors

    if operation == "backward":
        _prepare_backward(variant)
        grad_inputs, record = _measure_one(
            variant, "backward",
            lambda: layer.backward(variant["grad_output"]),
            capture_events=capture_events,
        )
        errors = {}
        if verify:
            errors = {
                "grad_weights": _max_abs_error(
                    variant["backend"].buffer_to_host(layer.grad_weights),
                    expected["grad_weights"], "grad_weights",
                ),
                "grad_bias": _max_abs_error(
                    variant["backend"].buffer_to_host(layer.grad_bias),
                    expected["grad_bias"], "grad_bias",
                ),
                "grad_inputs": _max_abs_error(
                    variant["backend"].buffer_to_host(grad_inputs),
                    expected["grad_inputs"], "grad_inputs",
                ),
            }
        grad_inputs.close()
        _close_if_open(layer.output)
        return record, errors

    if operation == "step":
        _reset_parameters(variant, case)
        _prepare_backward(variant)
        grad_inputs = layer.backward(variant["grad_output"])
        _, record = _measure_one(
            variant, "step",
            lambda: layer.step(LEARNING_RATE),
            capture_events=capture_events,
        )
        errors = {}
        if verify:
            errors = {
                "weights": _max_abs_error(
                    variant["backend"].buffer_to_host(layer.weights),
                    expected["weights"], "step weights",
                ),
                "bias": _max_abs_error(
                    variant["backend"].buffer_to_host(layer.bias),
                    expected["bias"], "step bias",
                ),
            }
        grad_inputs.close()
        _close_if_open(layer.output)
        return record, errors

    raise ValueError(f"unsupported profiling operation: {operation}")


def _summary_operation(
    operation: str,
    variants: dict[str, dict[str, Any]],
    case: dict[str, Any],
    expected: dict[str, Any],
    expected_cycles: list[dict[str, Any]],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    if operation == "resident_cycle":
        for variant in variants.values():
            _reset_parameters(variant, case)
    results = {
        condition: {"samples": [], "numerical_max_abs_error": {}}
        for condition in ("profiled", "control")
    }
    expected_by_cycle = {item["cycle_index"]: item for item in expected_cycles}

    for index in range(warmups + repetitions):
        order = ("profiled", "control") if index % 2 == 0 else ("control", "profiled")
        for condition in order:
            cycle_expected = expected_by_cycle.get(index)
            sample, errors = _measure_operation_once(
                variants[condition],
                operation,
                case,
                expected,
                cycle_expected,
                index,
                warmups,
                repetitions,
            )
            if condition == "profiled":
                expected_event_count = 3 if operation == "resident_cycle" else 1
                if len(sample["events"]) != expected_event_count:
                    raise RuntimeError(
                        f"{operation} expected {expected_event_count} events, "
                        f"captured {len(sample['events'])}"
                    )
            if index >= warmups:
                sample["sample_index"] = index - warmups
                results[condition]["samples"].append(sample)
                if errors:
                    results[condition]["numerical_max_abs_error"][str(index)] = errors

    for condition in ("profiled", "control"):
        samples = results[condition]["samples"]
        results[condition]["host_operation_stats_ns"] = _stats(
            [sample["host_operation_ns"] for sample in samples]
        )
        results[condition]["linear_array_validation_stats_ns"] = _stats(
            [sample["linear_array_validation_ns"] for sample in samples]
        )
        results[condition]["unattributed_python_and_layer_stats_ns"] = _stats(
            [sample["unattributed_python_and_layer_ns"] for sample in samples]
        )

    profiled_samples = results["profiled"]["samples"]
    control_samples = results["control"]["samples"]
    paired_delta = [
        left["host_operation_ns"] - right["host_operation_ns"]
        for left, right in zip(profiled_samples, control_samples)
    ]
    paired_percent = [
        100.0 * delta / control["host_operation_ns"]
        for delta, control in zip(paired_delta, control_samples)
        if control["host_operation_ns"]
    ]
    results["profiled_minus_unprofiled_host_ns"] = {
        "paired_delta_stats_ns": _stats(paired_delta),
        "paired_relative_delta_percent": _stats(paired_percent),
        "note": "Paired index; execution order alternates per repetition. Difference includes event creation/query/release and profiling-queue effects; it is not a universal profiler overhead constant.",
    }
    return results


def _transfer_trials(variant: dict[str, Any], case: dict[str, Any], *, warmups: int, repetitions: int) -> dict[str, Any]:
    api = variant["api"]
    backend = variant["backend"]
    flat = _flatten(case["weights"])
    host_source = (ctypes.c_double * len(flat))(*flat)
    host_dest = (ctypes.c_double * len(flat))()
    device_buffer = backend.memory.allocate(
        (case["output_size"], case["input_size"]), dtype="float64"
    )
    transfer_results: dict[str, Any] = {}
    try:
        # Prime the destination before measuring the blocking read path.
        api.write_buffer(
            backend.memory.queue, device_buffer.handle,
            ctypes.addressof(host_source), device_buffer.nbytes
        )
        for kind in ("h2d", "d2h"):
            samples: list[dict[str, Any]] = []
            for index in range(warmups + repetitions):
                if variant["profiled"]:
                    api.begin_capture(f"transfer_{kind}")
                started = time.perf_counter_ns()
                if kind == "h2d":
                    api.write_buffer(
                        backend.memory.queue, device_buffer.handle,
                        ctypes.addressof(host_source), device_buffer.nbytes
                    )
                else:
                    api.read_buffer(
                        backend.memory.queue, device_buffer.handle,
                        ctypes.addressof(host_dest), device_buffer.nbytes
                    )
                elapsed = time.perf_counter_ns() - started
                capture = api.end_capture() if variant["profiled"] else None
                if index >= warmups:
                    samples.append({
                        "host_blocking_call_ns": elapsed,
                        "events": capture["events"] if capture else [],
                        "host_api_components_ns": capture["host_components_ns"] if capture else {},
                    })
            if kind == "d2h":
                values = list(host_dest)
                transfer_results["d2h_roundtrip_max_abs_error"] = _max_abs_error(
                    values, flat, "transfer roundtrip"
                )
            transfer_results[kind] = {
                "bytes": device_buffer.nbytes,
                "samples": samples,
                "host_blocking_call_stats_ns": _stats(
                    [item["host_blocking_call_ns"] for item in samples]
                ),
            }
    finally:
        device_buffer.close()
    return transfer_results


def _worker_run(*, warmups: int, repetitions: int, worker_index: int) -> dict[str, Any]:
    case_reports = []
    system_before = _system_snapshot("0000:03:00.0")
    for case_index, (label, output_size, input_size) in enumerate(CASES):
        case = _case_data(output_size, input_size, seed=52052 + case_index)
        variants: dict[str, dict[str, Any]] = {}
        try:
            variants["profiled"] = _make_variant(case, profiled=True)
            variants["control"] = _make_variant(case, profiled=False)
            device = variants["profiled"]["device"]
            control_device = variants["control"]["device"]
            if device.pci_bdf != control_device.pci_bdf or device.name != control_device.name:
                raise RuntimeError("profiled and control APIs selected different devices")

            compile_times = {
                name: _compile_programs(variant)
                for name, variant in variants.items()
            }
            # Check initialization transfer fidelity and build CPU oracle values.
            cpu = CPUBackend()
            cpu_layer = Linear(input_size, output_size, seed=case["seed"], backend=cpu)
            initial_errors = {}
            for name, variant in variants.items():
                initial_errors[name] = {
                    "weights": _max_abs_error(
                        variant["backend"].buffer_to_host(variant["layer"].weights),
                        cpu_layer.weights, f"{name} initial weights",
                    ),
                    "bias": _max_abs_error(
                        variant["backend"].buffer_to_host(variant["layer"].bias),
                        cpu_layer.bias, f"{name} initial bias",
                    ),
                }
            cpu_layer.close()
            expected = _cpu_single_step(case)
            expected_cycles = _expected_cycle(
                case,
                warmups + repetitions,
                {warmups, warmups + repetitions - 1},
            )

            operations = {
                operation: _summary_operation(
                    operation,
                    variants,
                    case,
                    expected,
                    expected_cycles,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                for operation in ("forward", "backward", "step", "resident_cycle")
            }
            transfers = _transfer_trials(
                variants["profiled"], case, warmups=warmups, repetitions=repetitions
            )
            case_reports.append({
                "label": label,
                "weights_shape": [output_size, input_size],
                "seed": case["seed"],
                "dtype": "float64",
                "compile_api_elapsed_ns": compile_times,
                "variant_setup_elapsed_ns": {
                    name: {
                        "api_runtime_discovery_ns": variant["runtime_discovery_ns"],
                        "context_and_queue_creation_ns": variant["queue_creation_ns"],
                        "linear_parameters_and_resident_input_setup_ns": variant["layer_and_input_setup_ns"],
                    }
                    for name, variant in variants.items()
                },
                "profiling_queue_properties": variants["profiled"]["api"].queue_properties,
                "initial_parameter_max_abs_error": initial_errors,
                "operations": operations,
                "profiled_blocking_transfers": transfers,
                "expected_kernel_events_per_sample": {
                    "forward": 1, "backward": 1, "step": 1,
                    "resident_cycle": {"forward": 1, "backward": 1, "step": 1},
                },
            })
        finally:
            for variant in variants.values():
                for value in (variant.get("layer"), variant.get("inputs"), variant.get("grad_output")):
                    if value is not None:
                        value.close()
                variant["backend"].close()

    reference_device = _device_record(variants["profiled"]["device"])
    return {
        "worker_index": worker_index,
        "pid": os.getpid(),
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu_model": _cpu_model(),
            "cpu_logical_processors": os.cpu_count(),
            "device": reference_device,
            "rusticl_enable": os.environ.get("RUSTICL_ENABLE"),
            "rusticl_features": os.environ.get("RUSTICL_FEATURES"),
            "runtime_warning": "Patched Mesa libclc not detected",
        },
        "system_before": system_before,
        "system_after": _system_snapshot("0000:03:00.0"),
        "cases": case_reports,
    }


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or None
    for line in text.splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _source_fingerprint() -> str:
    paths = [
        "src/kinera/backends/base.py",
        "src/kinera/backends/cpu/backend.py",
        "src/kinera/nn/layers/linear.py",
        "src/kinera/backends/opencl/api.py",
        "src/kinera/backends/opencl/backend.py",
        "src/kinera/backends/opencl/buffer.py",
        "src/kinera/backends/opencl/memory.py",
        "src/kinera/backends/opencl/runtime.py",
        "src/kinera/backends/opencl/kernels/linear_forward.cl",
        "src/kinera/backends/opencl/kernels/linear_backward.cl",
        "src/kinera/backends/opencl/kernels/linear_step.cl",
        "experiments/benchmark_opencl_linear.py",
        "experiments/profile_opencl_device.py",
    ]
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _aggregate(workers: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for case_index, (label, _output_size, _input_size) in enumerate(CASES):
        case_aggregate: dict[str, Any] = {"operations": {}}
        for operation in ("forward", "backward", "step", "resident_cycle"):
            op_summary: dict[str, Any] = {"conditions": {}}
            for condition in ("profiled", "control"):
                samples = [
                    sample
                    for worker in workers
                    for sample in worker["cases"][case_index]["operations"][operation][condition]["samples"]
                ]
                process_medians = [
                    worker["cases"][case_index]["operations"][operation][condition]["host_operation_stats_ns"]["median"]
                    for worker in workers
                ]
                op_summary["conditions"][condition] = {
                    "host_operation_ns": _stats([sample["host_operation_ns"] for sample in samples]),
                    "process_median_ns": _stats(process_medians),
                    "linear_array_validation_ns": _stats([sample["linear_array_validation_ns"] for sample in samples]),
                    "unattributed_python_and_layer_ns": _stats([sample["unattributed_python_and_layer_ns"] for sample in samples]),
                }
                component_names = sorted({
                    name
                    for sample in samples
                    for name in sample["host_api_components_ns"]
                })
                op_summary["conditions"][condition]["host_api_components_ns"] = {
                    name: _stats([
                        sum(sample["host_api_components_ns"].get(name, []))
                        for sample in samples
                    ])
                    for name in component_names
                }
                if operation == "resident_cycle":
                    op_summary["conditions"][condition]["public_linear_stage_ns"] = {
                        stage: _stats([
                            sample["public_linear_stage_ns"][stage]
                            for sample in samples
                        ])
                        for stage in ("forward", "backward", "step")
                    }

            profiled = op_summary["conditions"]["profiled"]["host_operation_ns"]
            control = op_summary["conditions"]["control"]["host_operation_ns"]
            deltas = [
                sample["host_operation_ns"] - control_sample["host_operation_ns"]
                for worker in workers
                for sample, control_sample in zip(
                    worker["cases"][case_index]["operations"][operation]["profiled"]["samples"],
                    worker["cases"][case_index]["operations"][operation]["control"]["samples"],
                )
            ]
            event_stages = ("forward", "backward", "step") if operation == "resident_cycle" else (operation,)
            event_summary = {}
            for stage in event_stages:
                durations = [
                    event["durations_ns"]["kernel_execution_ns"]
                    for worker in workers
                    for sample in worker["cases"][case_index]["operations"][operation]["profiled"]["samples"]
                    for event in sample["events"]
                    if event["stage"] == stage and event.get("profiling_available")
                ]
                process_medians = []
                for worker in workers:
                    process_values = [
                        event["durations_ns"]["kernel_execution_ns"]
                        for sample in worker["cases"][case_index]["operations"][operation]["profiled"]["samples"]
                        for event in sample["events"]
                        if event["stage"] == stage and event.get("profiling_available")
                    ]
                    if process_values:
                        process_medians.append(statistics.median(process_values))
                queued = [
                    event["durations_ns"]["queued_to_submit_ns"]
                    for worker in workers
                    for sample in worker["cases"][case_index]["operations"][operation]["profiled"]["samples"]
                    for event in sample["events"]
                    if event["stage"] == stage and event.get("profiling_available")
                ]
                submit_start = [
                    event["durations_ns"]["submit_to_start_ns"]
                    for worker in workers
                    for sample in worker["cases"][case_index]["operations"][operation]["profiled"]["samples"]
                    for event in sample["events"]
                    if event["stage"] == stage and event.get("profiling_available")
                ]
                event_summary[stage] = {
                    "kernel_execution_ns": _stats(durations),
                    "process_median_ns": _stats(process_medians),
                    "queued_to_submit_ns": _stats(queued),
                    "submit_to_start_ns": _stats(submit_start),
                    "valid_event_samples": len(durations),
                }
            op_summary["paired_profiled_minus_control_host_ns"] = _stats(deltas)
            op_summary["device_events"] = event_summary
            op_summary["note"] = "Device and host timestamp domains are separate. Event execution is not added to clFinish or operation wall time."
            case_aggregate["operations"][operation] = op_summary

        case_aggregate["profiled_blocking_transfers"] = {}
        for kind in ("h2d", "d2h"):
            samples = [
                sample
                for worker in workers
                for sample in worker["cases"][case_index]["profiled_blocking_transfers"][kind]["samples"]
            ]
            durations = [
                event["durations_ns"]["command_execution_ns"]
                for sample in samples for event in sample["events"]
                if event.get("profiling_available")
            ]
            case_aggregate["profiled_blocking_transfers"][kind] = {
                "host_blocking_call_ns": _stats([sample["host_blocking_call_ns"] for sample in samples]),
                "device_event_execution_ns": _stats(durations),
                "process_median_device_event_ns": _stats([
                    statistics.median([
                        event["durations_ns"]["command_execution_ns"]
                        for sample in worker["cases"][case_index]["profiled_blocking_transfers"][kind]["samples"]
                        for event in sample["events"]
                        if event.get("profiling_available")
                    ])
                    for worker in workers
                    if any(
                        event.get("profiling_available")
                        for sample in worker["cases"][case_index]["profiled_blocking_transfers"][kind]["samples"]
                        for event in sample["events"]
                    )
                ]),
                "valid_event_samples": len(durations),
            }
        case_aggregate["setup"] = {
            "api_runtime_discovery_ns": {
                condition: _stats([
                    worker["cases"][case_index]["variant_setup_elapsed_ns"][condition]["api_runtime_discovery_ns"]
                    for worker in workers
                ])
                for condition in ("profiled", "control")
            },
            "context_and_queue_creation_ns": {
                condition: _stats([
                    worker["cases"][case_index]["variant_setup_elapsed_ns"][condition]["context_and_queue_creation_ns"]
                    for worker in workers
                ])
                for condition in ("profiled", "control")
            },
            "linear_parameters_and_input_setup_ns": {
                condition: _stats([
                    worker["cases"][case_index]["variant_setup_elapsed_ns"][condition]["linear_parameters_and_resident_input_setup_ns"]
                    for worker in workers
                ])
                for condition in ("profiled", "control")
            },
            "program_build_end_to_end_ns": {
                condition: {
                    operation: _stats([
                        worker["cases"][case_index]["compile_api_elapsed_ns"][condition][operation]
                        for worker in workers
                    ])
                    for operation in ("forward", "backward", "step")
                }
                for condition in ("profiled", "control")
            },
        }
        aggregate[label] = case_aggregate
    return aggregate


def _run_parent(*, warmups: int, repetitions: int, process_runs: int) -> dict[str, Any]:
    workers = []
    for worker_index in range(process_runs):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--warmups", str(warmups),
            "--repetitions", str(repetitions),
            "--worker-index", str(worker_index),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"P3e worker {worker_index} failed with status {completed.returncode}: "
                f"{completed.stderr[-4000:]}"
            )
        workers.append(json.loads(completed.stdout))
    return {
        "schema_version": 1,
        "phase": "issue-52-p3e-opencl-device-profiling",
        "code": {
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ).strip(),
            "source_fingerprint_sha256": _source_fingerprint(),
        },
        "protocol": {
            "process_runs": process_runs,
            "warmups_per_operation": warmups,
            "timed_repetitions_per_operation": repetitions,
            "process_order": "workers run sequentially; profiled/control order alternates per paired repetition",
            "shapes": [{"label": label, "weights_shape": [out, inp], "seed": 52052 + index} for index, (label, out, inp) in enumerate(CASES)],
            "dtype": "float64",
            "atol": ATOL,
            "rtol": RTOL,
            "learning_rate": LEARNING_RATE,
            "device_queue": "CL_QUEUE_PROFILING_ENABLE (verified with clGetCommandQueueInfo)",
            "event_profile_fields": ["CL_PROFILING_COMMAND_QUEUED", "CL_PROFILING_COMMAND_SUBMIT", "CL_PROFILING_COMMAND_START", "CL_PROFILING_COMMAND_END"],
            "production_sync_preserved": "Kernel wrapper still calls clFinish once after each successful enqueue; blocking H2D/D2H retain CL_TRUE semantics.",
            "host_timer": "time.perf_counter_ns; host timestamps are never subtracted from device timestamps.",
            "event_lifecycle": "Each successful enqueue event is queried after command completion and released once in a finally-safe path; failed profiling reads are recorded as unavailable.",
            "instrumentation_overhead": "Profiled and normal queues run paired, sequential, alternating-order samples. Host delta includes profiling queue/event creation, query/release, and driver effects; no fixed correction is applied.",
            "comparison_to_p3": "Same case seeds and data generator as experiments/benchmark_opencl_linear.py; P3 median wall times are context only and are not pooled with P3e samples.",
            "concurrency": "One worker process and one GPU command at a time; workers are sequential. Before/after CPU load, GPU busy, clocks and VRAM are recorded when exposed. No P3d, training, or historical pilot is launched.",
            "numerical_tolerances": {"atol": ATOL, "rtol": RTOL},
        },
        "workers": workers,
        "aggregate": _aggregate(workers),
        "limitations": [
            "Profiling timestamps are reported as device-clock intervals; they are not on the perf_counter_ns host clock.",
            "clFinish host wait includes runtime/driver work, queued commands and completion wait; it is not assigned to kernel execution.",
            "Kernel execution time is the OpenCL event END minus START interval; event queue intervals are reported separately.",
            "Adding events, profiling-enabled queues, four profiling queries and event release changes host overhead. The paired plain-queue control measures the combined effect under this protocol, not a portable correction factor.",
            "The queue is in-order and each production API kernel call synchronizes with clFinish; the integrated cycle contains separately profiled forward, backward and step events.",
            "P3's CPU/OpenCL measurements used host wall timing and a previous baseline commit; they are contextual only and are not direct paired comparisons to P3e.",
            "Driver/compiler cache, clocks, thermal state and external workloads are not locked. Hardware utilization snapshots are observational and may miss short concurrent activity.",
            "Rusticl's 'Patched Mesa libclc not detected' warning remains; this experiment verifies event profiling behavior on this runtime build only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--process-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 5 or args.process_runs < 2:
        parser.error("require warmups >= 1, repetitions >= 5, process-runs >= 2")
    if args.worker:
        result = _worker_run(
            warmups=args.warmups,
            repetitions=args.repetitions,
            worker_index=args.worker_index,
        )
    else:
        result = _run_parent(
            warmups=args.warmups,
            repetitions=args.repetitions,
            process_runs=args.process_runs,
        )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
