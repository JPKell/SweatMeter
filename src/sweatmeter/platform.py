"""Composition boundary for host and GPU telemetry implementations.

This is the only module outside the platform readers that branches on ``sys.platform``. Tier-3
platforms return explicit stubs, unknown platforms raise a catchable suite error, and callers that
choose to degrade can install the public null implementations without fabricating zero readings.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from enum import StrEnum

from baseaicore import (
    UNSUPPORTED,
    GpuProfile,
    Measurement,
    UnsupportedPlatformError,
    ValidationError,
)

from sweatmeter.readers.darwin import DarwinHostReader
from sweatmeter.readers.linux import LinuxHostReader
from sweatmeter.readers.nvidia import NvidiaSmiReader
from sweatmeter.readers.protocols import GpuReader, HostReader
from sweatmeter.readers.windows import WindowsHostReader
from sweatmeter.types import DiskThroughput, GpuSample, HostFacts, MemoryReading

__all__ = [
    "GpuBackend",
    "NullGpuReader",
    "NullHostReader",
    "create_gpu_reader",
    "create_host_reader",
]

_HOST_LIVE_FIELDS = (
    "cpu_percent",
    "load_average_1m",
    "ram_used_bytes",
    "ram_available_bytes",
    "ram_total_bytes",
    "cpu_temperature_c",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "process_rss_bytes",
)
_HOST_STATIC_FIELDS = (
    "hostname",
    "os_name",
    "os_version",
    "kernel",
    "architecture",
    "cpu_model",
    "physical_cores",
    "logical_cores",
    "ram_bytes",
)


class GpuBackend(StrEnum):
    """GPU collection backend selectable by :func:`create_gpu_reader`.

    NVIDIA's bounded command interface is the only Phase 3 backend. Naming it now makes later
    backends additive instead of changing the factory contract.
    """

    NVIDIA_SMI = "nvidia-smi"


class NullHostReader:
    """Provide an intentionally unavailable host reader for graceful degradation.

    Every measurement is ``UNSUPPORTED`` and every field carries the configured reason. The class
    is safe for concurrent calls because it retains no sampling state.

    Args:
        reason: Machine-readable explanation returned for every unavailable field.

    Raises:
        ValidationError: If ``reason`` is empty or only whitespace.
    """

    def __init__(self, *, reason: str = "platform_unsupported") -> None:
        """Configure the reason attached to every unavailable host field."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationError(
                f"NullHostReader.reason must be non-empty; got {reason!r}.",
                details={"field": "reason", "value": reason},
            )
        self._reason = reason.strip()

    def cpu_percent(self) -> Measurement:
        """Return ``UNSUPPORTED`` because this reader performs no host collection."""
        return UNSUPPORTED

    def load_average_1m(self) -> Measurement:
        """Return ``UNSUPPORTED`` because this reader performs no host collection."""
        return UNSUPPORTED

    def memory(self) -> MemoryReading:
        """Return wholly unsupported RAM measurements."""
        return MemoryReading()

    def cpu_temperature(self) -> Measurement:
        """Return ``UNSUPPORTED`` because this reader performs no host collection."""
        return UNSUPPORTED

    def disk_throughput(self) -> DiskThroughput:
        """Return wholly unsupported disk-throughput measurements."""
        return DiskThroughput()

    def process_rss_bytes(self) -> Measurement:
        """Return ``UNSUPPORTED`` because this reader performs no process collection."""
        return UNSUPPORTED

    def static_facts(self) -> HostFacts:
        """Return wholly unsupported static host facts."""
        return HostFacts()

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return the configured reason for every host field this reader cannot provide."""
        return dict.fromkeys((*_HOST_LIVE_FIELDS, *_HOST_STATIC_FIELDS), self._reason)


class NullGpuReader:
    """Provide an intentionally unavailable GPU reader without inventing devices.

    ``available()`` is false and both live and static collections are empty. The diagnostic reason
    lets a collector distinguish deliberate degradation from a real zero-GPU probe.

    Args:
        reason: Machine-readable explanation for GPU unavailability.

    Raises:
        ValidationError: If ``reason`` is empty or only whitespace.
    """

    def __init__(self, *, reason: str = "platform_unsupported") -> None:
        """Configure the reason attached to GPU unavailability."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationError(
                f"NullGpuReader.reason must be non-empty; got {reason!r}.",
                details={"field": "reason", "value": reason},
            )
        self._reason = reason.strip()

    def available(self) -> bool:
        """Return false because this reader intentionally exposes no GPU backend."""
        return False

    def sample(self) -> tuple[GpuSample, ...]:
        """Return no live GPU samples."""
        return ()

    def static_info(self) -> tuple[GpuProfile, ...]:
        """Return no static GPU profiles."""
        return ()

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return the configured reason for GPU unavailability."""
        return {"gpu": self._reason}


def create_host_reader(*, platform_name: str | None = None) -> HostReader:
    """Return the host reader selected for an explicitly named or current platform.

    Linux receives its implemented reader. Windows and macOS receive tier-3 stubs whose methods
    raise ``UnsupportedPlatformError``; returning the stub keeps selection separate from use and
    makes the unsupported surface directly testable.

    Args:
        platform_name: ``sys.platform``-style name. Defaults to the running interpreter's value.

    Returns:
        The platform's implemented reader or documented tier-3 stub.

    Raises:
        UnsupportedPlatformError: If the platform has no recognized reader or stub.
    """
    selected = sys.platform if platform_name is None else platform_name
    match selected:
        case "linux":
            return LinuxHostReader()
        case "win32":
            return WindowsHostReader()
        case "darwin":
            return DarwinHostReader()
        case other:
            raise UnsupportedPlatformError(
                f"Host telemetry is not implemented for platform {other!r}; construct "
                "NullHostReader() to continue with unsupported measurements.",
                details={"platform": other, "feature": "host telemetry"},
            )


def create_gpu_reader(*, prefer: GpuBackend | None = None) -> GpuReader:
    """Return the requested GPU reader without probing hardware during construction.

    Availability remains a live reader operation: an absent or failing ``nvidia-smi`` degrades on
    every call and is retried on the next one.

    Args:
        prefer: Backend preference. ``None`` selects Phase 3's NVIDIA command backend.

    Returns:
        A freshly configured GPU reader.

    Raises:
        ValidationError: If a value outside the supported backend enum is supplied at runtime.
    """
    if prefer is None or prefer is GpuBackend.NVIDIA_SMI:
        return NvidiaSmiReader()
    raise ValidationError(
        f"Unsupported GPU backend {prefer!r}; expected {GpuBackend.NVIDIA_SMI.value!r}.",
        details={"field": "prefer", "value": str(prefer)},
    )
