"""Tier-3 Windows host-reader stub with explicit, catchable failure."""

from __future__ import annotations

from typing import NoReturn

from baseaicore import Measurement, UnsupportedPlatformError

from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading

__all__ = ["WindowsHostReader"]


class WindowsHostReader:
    """Represent the unimplemented Windows host telemetry interface.

    Every method raises ``UnsupportedPlatformError`` rather than returning plausible-looking zero
    values. Callers that can degrade should catch the error and use ``NullHostReader``. The stub
    has no mutable state and is safe to share between threads.
    """

    @staticmethod
    def _unsupported() -> NoReturn:
        """Raise the shared error for the tier-3 implementation gap."""
        raise UnsupportedPlatformError(
            "Windows host telemetry is a tier-3 interface only; PDH/WMI readers are not "
            "implemented. Use NullHostReader() to degrade.",
            details={"platform": "win32", "feature": "host telemetry"},
        )

    def cpu_percent(self) -> Measurement:
        """Raise because Windows CPU telemetry is not implemented."""
        self._unsupported()

    def load_average_1m(self) -> Measurement:
        """Raise because Windows load telemetry is not implemented."""
        self._unsupported()

    def memory(self) -> MemoryReading:
        """Raise because Windows memory telemetry is not implemented."""
        self._unsupported()

    def cpu_temperature(self) -> Measurement:
        """Raise because Windows thermal telemetry is not implemented."""
        self._unsupported()

    def disk_throughput(self) -> DiskThroughput:
        """Raise because Windows disk telemetry is not implemented."""
        self._unsupported()

    def process_rss_bytes(self) -> Measurement:
        """Raise because Windows process telemetry is not implemented."""
        self._unsupported()

    def static_facts(self) -> HostFacts:
        """Raise because Windows static profiling is not implemented."""
        self._unsupported()
