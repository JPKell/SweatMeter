"""Structural interfaces implemented by platform-specific telemetry readers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from baseaicore import GpuProfile, Measurement

from sweatmeter.types import DiskThroughput, GpuSample, HostFacts, MemoryReading

__all__ = ["GpuReader", "HostReader"]


@runtime_checkable
class HostReader(Protocol):
    """Host telemetry surface consumed by the future collector.

    Implementations isolate platform-specific access. Linux uses ``/proc`` and ``/sys``; future
    Windows and macOS readers can satisfy this protocol without changing collector code.
    """

    def cpu_percent(self) -> Measurement:
        """Return aggregate CPU utilization since the preceding call."""
        ...

    def load_average_1m(self) -> Measurement:
        """Return the one-minute load average."""
        ...

    def memory(self) -> MemoryReading:
        """Return total, available, and used RAM in bytes."""
        ...

    def cpu_temperature(self) -> Measurement:
        """Return a CPU temperature in degrees Celsius."""
        ...

    def disk_throughput(self) -> DiskThroughput:
        """Return whole-device read and write throughput since the preceding call."""
        ...

    def process_rss_bytes(self) -> Measurement:
        """Return the current process resident-set size in bytes."""
        ...

    def static_facts(self) -> HostFacts:
        """Return static identity and capacity facts for this host."""
        ...


@runtime_checkable
class GpuReader(Protocol):
    """GPU telemetry surface consumed by the future collector."""

    def available(self) -> bool:
        """Return whether at least one GPU can be queried now."""
        ...

    def sample(self) -> Sequence[GpuSample]:
        """Return one live sample for every visible GPU."""
        ...

    def static_info(self) -> Sequence[GpuProfile]:
        """Return static identity and capacity information for every visible GPU."""
        ...
