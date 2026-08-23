"""Tier-3 macOS host-reader stub with explicit, catchable failure."""

from __future__ import annotations

from typing import NoReturn

from baseaicore import Measurement, UnsupportedPlatformError

from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading

__all__ = ["DarwinHostReader"]


class DarwinHostReader:
    """Represent the unimplemented macOS host telemetry interface.

    Every method raises ``UnsupportedPlatformError`` rather than returning plausible-looking zero
    values. Callers that can degrade should catch the error and use ``NullHostReader``. The stub
    has no mutable state and is safe to share between threads.
    """

    @staticmethod
    def _unsupported() -> NoReturn:
        """Raise the shared error for the tier-3 implementation gap."""
        raise UnsupportedPlatformError(
            "macOS host telemetry is a tier-3 interface only; sysctl/SMC readers are not "
            "implemented. Use NullHostReader() to degrade.",
            details={"platform": "darwin", "feature": "host telemetry"},
        )

    def cpu_percent(self) -> Measurement:
        """Raise because macOS CPU telemetry is not implemented."""
        self._unsupported()

    def load_average_1m(self) -> Measurement:
        """Raise because macOS load telemetry is not implemented."""
        self._unsupported()

    def memory(self) -> MemoryReading:
        """Raise because macOS memory telemetry is not implemented."""
        self._unsupported()

    def cpu_temperature(self) -> Measurement:
        """Raise because macOS thermal telemetry is not implemented."""
        self._unsupported()

    def disk_throughput(self) -> DiskThroughput:
        """Raise because macOS disk telemetry is not implemented."""
        self._unsupported()

    def process_rss_bytes(self) -> Measurement:
        """Raise because macOS process telemetry is not implemented."""
        self._unsupported()

    def static_facts(self) -> HostFacts:
        """Raise because macOS static profiling is not implemented."""
        self._unsupported()
