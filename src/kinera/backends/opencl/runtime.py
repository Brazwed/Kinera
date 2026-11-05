"""Lazy OpenCL platform and GPU capability discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kinera.backends.base import BackendError, BackendUnavailableError


OPENCL_DEVICE_TYPE_GPU = 1 << 2


@dataclass(frozen=True, slots=True)
class OpenCLDeviceInfo:
    """OpenCL GPU identity and the limits relevant to Kinera buffers."""

    index: int
    handle: int
    name: str
    vendor: str
    vendor_id: int
    device_type: int
    platform_name: str
    platform_vendor: str
    version: str
    opencl_c_version: str
    driver_version: str
    extensions: tuple[str, ...]
    pci_bus_info: tuple[int, int, int, int] | None
    global_mem_size: int
    max_mem_alloc_size: int
    local_mem_size: int
    max_work_group_size: int
    max_work_item_sizes: tuple[int, ...]
    double_fp_config: int
    available: bool
    compiler_available: bool

    @property
    def supports_fp64(self) -> bool:
        has_extension = "cl_khr_fp64" in self.extensions or "cl_amd_fp64" in self.extensions
        return has_extension and self.double_fp_config != 0

    @property
    def pci_bdf(self) -> str | None:
        if self.pci_bus_info is None:
            return None
        domain, bus, device, function = self.pci_bus_info
        return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


class OpenCLRuntime:
    """Discover GPU devices on explicit OpenCL-backend construction only."""

    def __init__(self, api: Any | None = None) -> None:
        if api is None:
            try:
                from kinera.backends.opencl.api import OpenCLApi

                api = OpenCLApi()
            except BackendUnavailableError as error:
                self.api = None
                self._devices: list[OpenCLDeviceInfo] = []
                self._last_error = str(error)
                return

        self.api = api
        self._devices = []
        self._last_error: str | None = None
        try:
            self._devices = self._discover_devices()
            if not self._devices:
                self._last_error = "OpenCL found no GPU devices"
        except BackendError as error:
            self._last_error = str(error)
            self._devices = []

    @property
    def is_available(self) -> bool:
        return bool(self._devices)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def device_count(self) -> int:
        return len(self._devices)

    def devices(self) -> list[OpenCLDeviceInfo]:
        return list(self._devices)

    def require_device(
        self, device_id: int = 0, *, require_fp64: bool = False
    ) -> OpenCLDeviceInfo:
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError("OpenCL device_id must be an integer")
        if not self._devices:
            detail = f": {self._last_error}" if self._last_error else ""
            raise BackendUnavailableError(f"OpenCL GPU is unavailable{detail}")
        if device_id < 0 or device_id >= len(self._devices):
            raise BackendUnavailableError(
                f"OpenCL GPU {device_id} is unavailable; detected "
                f"{len(self._devices)} GPU device(s)"
            )

        device = self._devices[device_id]
        if not device.available:
            raise BackendUnavailableError(
                f"OpenCL GPU {device_id} ({device.name}) is not available"
            )
        if not device.compiler_available:
            raise BackendUnavailableError(
                f"OpenCL compiler is unavailable for GPU {device_id} ({device.name})"
            )
        if require_fp64 and not device.supports_fp64:
            raise BackendUnavailableError(
                f"OpenCL GPU {device_id} ({device.name}) does not expose the required "
                "cl_khr_fp64/cl_amd_fp64 extension and nonzero "
                "CL_DEVICE_DOUBLE_FP_CONFIG; for Mesa Rusticl, enable "
                "RUSTICL_FEATURES=fp64 before starting Python"
            )
        return device

    def _discover_devices(self) -> list[OpenCLDeviceInfo]:
        platforms = self.api.platforms()
        devices: list[OpenCLDeviceInfo] = []
        for platform in platforms:
            platform_info = self.api.platform_info(platform)
            for handle in self.api.devices(platform):
                raw = self.api.device_info(handle, platform_info)
                device_type = int(raw["device_type"])
                if not device_type & OPENCL_DEVICE_TYPE_GPU:
                    continue
                pci_raw = raw.get("pci_bus_info")
                pci = (
                    (
                        int(pci_raw["domain"]),
                        int(pci_raw["bus"]),
                        int(pci_raw["device"]),
                        int(pci_raw["function"]),
                    )
                    if pci_raw is not None
                    else None
                )
                extensions = tuple(str(value) for value in raw["extensions"])
                devices.append(
                    OpenCLDeviceInfo(
                        index=len(devices),
                        handle=int(handle),
                        name=str(raw["name"]),
                        vendor=str(raw["vendor"]),
                        vendor_id=int(raw["vendor_id"]),
                        device_type=device_type,
                        platform_name=str(raw["platform_name"]),
                        platform_vendor=str(raw["platform_vendor"]),
                        version=str(raw["version"]),
                        opencl_c_version=str(raw["opencl_c_version"]),
                        driver_version=str(raw["driver_version"]),
                        extensions=extensions,
                        pci_bus_info=pci,
                        global_mem_size=int(raw["global_mem_size"]),
                        max_mem_alloc_size=int(raw["max_mem_alloc_size"]),
                        local_mem_size=int(raw["local_mem_size"]),
                        max_work_group_size=int(raw["max_work_group_size"]),
                        max_work_item_sizes=tuple(
                            int(value) for value in raw["max_work_item_sizes"]
                        ),
                        double_fp_config=int(raw["double_fp_config"]),
                        available=bool(raw["available"]),
                        compiler_available=bool(raw["compiler_available"]),
                    )
                )
        return devices
