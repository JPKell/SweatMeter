"""Immutable observations returned by SweatMeter readers and collectors.

The types in this module contain observations only. They do not read the host and they do not
silently replace an unavailable quantity with zero. Numeric fields therefore use
:class:`baseaicore.Measurement`, and unavailable text uses the same explicit
:data:`baseaicore.UNSUPPORTED` sentinel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from baseaicore import UNSUPPORTED, Measurement, StorageDevice, Unsupported, ValidationError

type ReportedText = str | Unsupported
"""Text reported by the operating system, or ``UNSUPPORTED`` when it could not be read."""

__all__ = [
    "DiskThroughput",
    "GpuSample",
    "HostFacts",
    "MemoryReading",
    "ReportedText",
    "TelemetrySnapshot",
]


@dataclass(frozen=True, slots=True)
class MemoryReading:
    """One reading of system memory, normalized to bytes.

    ``used_bytes`` is derived as ``total_bytes - available_bytes`` only when both inputs are
    trustworthy. A malformed or incomplete input affects only the fields that depend on it.
    """

    total_bytes: Measurement = UNSUPPORTED
    available_bytes: Measurement = UNSUPPORTED
    used_bytes: Measurement = UNSUPPORTED


@dataclass(frozen=True, slots=True)
class DiskThroughput:
    """Whole-device disk throughput between two samples, in bytes per second.

    The first reading is necessarily unsupported because cumulative kernel counters need a prior
    value and elapsed time before they can become a rate.
    """

    read_bytes_per_sec: Measurement = UNSUPPORTED
    write_bytes_per_sec: Measurement = UNSUPPORTED


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One live, per-device GPU telemetry reading in normalized units.

    The empty ``throttle_reasons`` tuple has two meanings: no queried reason is active, or the
    driver could not report reasons. ``throttle_reasons_available`` distinguishes those states.
    Static identity beyond ``uuid`` belongs in BaseAiCore's ``GpuProfile``, not this live value.
    """

    index: int
    uuid: str | None = None
    utilization_percent: Measurement = UNSUPPORTED
    memory_utilization_percent: Measurement = UNSUPPORTED
    vram_used_bytes: Measurement = UNSUPPORTED
    vram_total_bytes: Measurement = UNSUPPORTED
    temperature_c: Measurement = UNSUPPORTED
    memory_temperature_c: Measurement = UNSUPPORTED
    power_watts: Measurement = UNSUPPORTED
    power_limit_watts: Measurement = UNSUPPORTED
    fan_percent: Measurement = UNSUPPORTED
    core_clock_mhz: Measurement = UNSUPPORTED
    memory_clock_mhz: Measurement = UNSUPPORTED
    throttle_reasons: tuple[str, ...] = ()
    throttle_reasons_available: bool = False


@dataclass(frozen=True, slots=True)
class HostFacts:
    """Static host facts collected by a :class:`~sweatmeter.readers.HostReader`.

    Live utilization never appears here. These values are later assembled into BaseAiCore's
    ``MachineProfile`` by the Phase 3 collector. Storage is retained as provenance but is excluded
    from BaseAiCore's machine fingerprint.
    """

    hostname: ReportedText = UNSUPPORTED
    os_name: ReportedText = UNSUPPORTED
    os_version: ReportedText = UNSUPPORTED
    kernel: ReportedText = UNSUPPORTED
    architecture: ReportedText = UNSUPPORTED
    cpu_model: ReportedText = UNSUPPORTED
    physical_cores: Measurement = UNSUPPORTED
    logical_cores: Measurement = UNSUPPORTED
    ram_bytes: Measurement = UNSUPPORTED
    storage: tuple[StorageDevice, ...] = ()
    python_version: ReportedText = UNSUPPORTED


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Represent one complete, timezone-aware observation of live machine telemetry.

    Static identity is deliberately absent; it belongs in BaseAiCore's ``MachineProfile``. Missing
    measurements remain ``UNSUPPORTED`` and their machine-readable explanations are available from
    :meth:`unavailable_reasons`. The diagnostic tuple is immutable internally, and callers receive
    a copy so a snapshot remains safe to share between sampler and consumer threads.

    Attributes:
        timestamp: Timezone-aware instant at which collection began.
        cpu_percent: Aggregate host CPU utilization from 0 through 100.
        load_average_1m: Host one-minute load average.
        ram_used_bytes: Currently used host RAM.
        ram_available_bytes: RAM available without swapping.
        ram_total_bytes: Installed host RAM visible to the operating system.
        cpu_temperature_c: CPU or package temperature in degrees Celsius.
        disk_read_bytes_per_sec: Whole-device read throughput.
        disk_write_bytes_per_sec: Whole-device write throughput.
        process_rss_bytes: Current resident-set size of this Python process.
        gpus: Per-device GPU observations; no cross-device aggregate is created.

    Raises:
        ValidationError: If ``timestamp`` is naive. A snapshot without an absolute instant cannot
            expose staleness or support time-based integration safely.
    """

    timestamp: datetime
    cpu_percent: Measurement = UNSUPPORTED
    load_average_1m: Measurement = UNSUPPORTED
    ram_used_bytes: Measurement = UNSUPPORTED
    ram_available_bytes: Measurement = UNSUPPORTED
    ram_total_bytes: Measurement = UNSUPPORTED
    cpu_temperature_c: Measurement = UNSUPPORTED
    disk_read_bytes_per_sec: Measurement = UNSUPPORTED
    disk_write_bytes_per_sec: Measurement = UNSUPPORTED
    process_rss_bytes: Measurement = UNSUPPORTED
    gpus: tuple[GpuSample, ...] = ()
    _reasons: tuple[tuple[str, str], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject timestamps whose absolute instant cannot be established."""
        tzinfo = self.timestamp.tzinfo
        if tzinfo is None or tzinfo.utcoffset(self.timestamp) is None:
            raise ValidationError(
                "TelemetrySnapshot.timestamp must be timezone-aware; a naive timestamp cannot "
                "show sample age reliably.",
                details={"field": "timestamp"},
            )

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return a copy of the reason recorded for every unavailable snapshot field."""
        return dict(self._reasons)
