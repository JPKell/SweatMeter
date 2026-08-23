"""Shared fixtures, guards, and test doubles for portable SweatMeter tests."""

from __future__ import annotations

import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import pytest


@pytest.fixture
def telemetry_fixtures() -> Path:
    """Return the root of captured, platform-independent kernel-data fixtures."""
    return Path(__file__).parent / "fixtures" / "telemetry"


@pytest.fixture(autouse=True)
def _no_outbound_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a real network connection.

    SweatMeter promises no network access of any kind (spec §14), so an outbound connection from
    the default suite is a contract breach rather than a slow test. Live tests are exempt because
    they exercise real hardware tooling, per the suite testing standards.
    """
    if request.node.get_closest_marker("live") is not None:
        return

    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise RuntimeError(
            "SweatMeter tests must not open network connections; the package performs no network "
            "access and its default suite must pass with no network."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class NVMLError(Exception):  # mirrors the binding's class name exactly
    """Base for the fake binding's errors, matching pynvml's naming."""


class NVMLError_NotSupported(NVMLError):  # noqa: N801, N818 — the reason map keys on this name
    """The device does not implement this query."""


class NVMLError_NoPermission(NVMLError):  # noqa: N801, N818 — the reason map keys on this name
    """The caller may not perform this query."""


class NVMLError_DriverNotLoaded(NVMLError):  # noqa: N801, N818 — the reason map keys on this name
    """No NVIDIA driver is running."""


class NVMLError_GpuIsLost(NVMLError):  # noqa: N801, N818 — the reason map keys on this name
    """The device fell off the bus."""


@dataclass
class FakeDevice:
    """One device's answers to every NVML query the reader makes."""

    name: str | bytes = "NVIDIA Test GPU"
    uuid: str | bytes = "GPU-TEST-0"
    memory_used: int = 4_000_000_000
    memory_total: int = 16_000_000_000
    util_gpu: int = 50
    util_memory: int = 25
    temperature: int = 60
    power_milliwatts: int = 120_500
    power_limit_milliwatts: int = 180_000
    fan: int = 40
    clock_sm: int = 2_400
    clock_mem: int = 9_000
    compute_capability: tuple[int, int] = (12, 0)
    supported_throttle: int = 511
    current_throttle: int = 0


@dataclass
class _Memory:
    total: int
    used: int


@dataclass
class _Utilization:
    gpu: int
    memory: int


@dataclass
class FakeNvml:
    """A configurable stand-in for the NVML binding.

    ``fail`` maps an NVML function name to the exception it should raise, which is how a single
    unsupported sensor is simulated without touching any other reading.
    """

    devices: Sequence[FakeDevice] = field(default_factory=lambda: [FakeDevice()])
    fail: Mapping[str, Exception] = field(default_factory=dict)
    driver: str | bytes = "580.173.02"
    cuda: int = 13_000
    init_calls: int = 0
    shutdown_calls: int = 0

    NVML_TEMPERATURE_GPU: int = 0
    NVML_CLOCK_SM: int = 1
    NVML_CLOCK_MEM: int = 2

    def _check(self, operation: str) -> None:
        failure = self.fail.get(operation)
        if failure is not None:
            raise failure

    def _device(self, handle: object) -> FakeDevice:
        return self.devices[int(str(handle))]

    def nvmlInit(self) -> None:  # noqa: N802 — NVML C API name
        self._check("nvmlInit")
        self.init_calls += 1

    def nvmlShutdown(self) -> None:  # noqa: N802 — NVML C API name
        self._check("nvmlShutdown")
        self.shutdown_calls += 1

    def nvmlSystemGetDriverVersion(self) -> str | bytes:  # noqa: N802 — NVML C API name
        self._check("nvmlSystemGetDriverVersion")
        return self.driver

    def nvmlSystemGetCudaDriverVersion(self) -> int:  # noqa: N802 — NVML C API name
        self._check("nvmlSystemGetCudaDriverVersion")
        return self.cuda

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetCount")
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index: int, /) -> object:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetHandleByIndex")
        return str(index)

    def nvmlDeviceGetName(self, handle: object, /) -> str | bytes:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetName")
        return self._device(handle).name

    def nvmlDeviceGetUUID(self, handle: object, /) -> str | bytes:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetUUID")
        return self._device(handle).uuid

    def nvmlDeviceGetMemoryInfo(self, handle: object, /) -> object:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetMemoryInfo")
        device = self._device(handle)
        return _Memory(total=device.memory_total, used=device.memory_used)

    def nvmlDeviceGetUtilizationRates(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> object:
        self._check("nvmlDeviceGetUtilizationRates")
        device = self._device(handle)
        return _Utilization(gpu=device.util_gpu, memory=device.util_memory)

    def nvmlDeviceGetTemperature(  # noqa: N802 — NVML C API name
        self, handle: object, sensor: int, /
    ) -> int:
        self._check("nvmlDeviceGetTemperature")
        del sensor
        return self._device(handle).temperature

    def nvmlDeviceGetPowerUsage(self, handle: object, /) -> int:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetPowerUsage")
        return self._device(handle).power_milliwatts

    def nvmlDeviceGetEnforcedPowerLimit(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        self._check("nvmlDeviceGetEnforcedPowerLimit")
        return self._device(handle).power_limit_milliwatts

    def nvmlDeviceGetFanSpeed(self, handle: object, /) -> int:  # noqa: N802 — NVML C API name
        self._check("nvmlDeviceGetFanSpeed")
        return self._device(handle).fan

    def nvmlDeviceGetClockInfo(  # noqa: N802 — NVML C API name
        self, handle: object, clock: int, /
    ) -> int:
        self._check("nvmlDeviceGetClockInfo")
        device = self._device(handle)
        return device.clock_sm if clock == self.NVML_CLOCK_SM else device.clock_mem

    def nvmlDeviceGetCudaComputeCapability(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> tuple[int, int]:
        self._check("nvmlDeviceGetCudaComputeCapability")
        return self._device(handle).compute_capability

    def nvmlDeviceGetCurrentClocksThrottleReasons(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        self._check("nvmlDeviceGetCurrentClocksThrottleReasons")
        return self._device(handle).current_throttle

    def nvmlDeviceGetSupportedClocksThrottleReasons(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        self._check("nvmlDeviceGetSupportedClocksThrottleReasons")
        return self._device(handle).supported_throttle
