"""Structural interfaces implemented by platform-specific telemetry readers."""

from __future__ import annotations

from typing import Protocol

from baseaicore import Measurement

from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading

__all__ = ["HostReader"]


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

    def static_facts(self) -> HostFacts:
        """Return static identity and capacity facts for this host."""
        ...
