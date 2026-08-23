"""Supported deterministic readers for consumer and failure-path tests.

These doubles are package API, not test-suite helpers: downstream applications can exercise
admission, telemetry, and energy logic without monkeypatching operating-system readers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from baseaicore import UNSUPPORTED, GpuProfile, GpuVendor, Measurement, ValidationError

from sweatmeter.readers.protocols import GpuReader, HostReader
from sweatmeter.types import DiskThroughput, GpuSample, HostFacts, MemoryReading

__all__ = [
    "FaultInjectingReader",
    "HostReading",
    "ScriptedGpuReader",
    "ScriptedHostReader",
]

_HOST_OPERATIONS = frozenset(
    {
        "cpu_percent",
        "load_average_1m",
        "memory",
        "cpu_temperature",
        "disk_throughput",
        "process_rss_bytes",
        "static_facts",
    }
)
_GPU_OPERATIONS = frozenset({"available", "sample", "static_info"})


@dataclass(frozen=True, slots=True)
class HostReading:
    """Describe one deterministic host-reader step.

    Each field maps directly to one ``HostReader`` operation. Unsupported values are explicit and
    defaults make it easy to script only the metric a consumer cares about. Static facts are kept
    separate from live utilization inside the nested ``HostFacts`` value.
    """

    cpu_percent: Measurement = UNSUPPORTED
    load_average_1m: Measurement = UNSUPPORTED
    memory: MemoryReading = field(default_factory=MemoryReading)
    cpu_temperature_c: Measurement = UNSUPPORTED
    disk_throughput: DiskThroughput = field(default_factory=DiskThroughput)
    process_rss_bytes: Measurement = UNSUPPORTED
    static_facts: HostFacts = field(default_factory=HostFacts)


class ScriptedHostReader:
    """Replay a fixed host series deterministically, one call per operation per step.

    Every reader operation has its own cursor. A collector calling all operations once therefore
    receives one coherent script row, while focused consumer tests can call a single operation
    repeatedly without needing to call unrelated methods. Exhaustion raises ``IndexError`` so a
    test cannot silently reuse stale telemetry. Access is serialized and safe across threads.

    Args:
        samples: Fixed sequence copied during construction. An empty sequence is allowed and makes
            every operation immediately exhausted.
    """

    def __init__(self, samples: Sequence[HostReading]) -> None:
        """Copy the script and initialize one cursor per reader operation."""
        self._samples = tuple(samples)
        self._positions = dict.fromkeys(_HOST_OPERATIONS, 0)
        self._lock = threading.Lock()

    def _next(self, operation: str) -> HostReading:
        """Return and consume the next row for one reader operation."""
        with self._lock:
            position = self._positions[operation]
            if position >= len(self._samples):
                raise IndexError(f"ScriptedHostReader.{operation} exhausted at step {position}.")
            self._positions[operation] = position + 1
            return self._samples[position]

    def cpu_percent(self) -> Measurement:
        """Return the next scripted aggregate CPU utilization."""
        return self._next("cpu_percent").cpu_percent

    def load_average_1m(self) -> Measurement:
        """Return the next scripted one-minute load average."""
        return self._next("load_average_1m").load_average_1m

    def memory(self) -> MemoryReading:
        """Return the next scripted memory reading."""
        return self._next("memory").memory

    def cpu_temperature(self) -> Measurement:
        """Return the next scripted CPU temperature."""
        return self._next("cpu_temperature").cpu_temperature_c

    def disk_throughput(self) -> DiskThroughput:
        """Return the next scripted disk-throughput reading."""
        return self._next("disk_throughput").disk_throughput

    def process_rss_bytes(self) -> Measurement:
        """Return the next scripted process resident-set size."""
        return self._next("process_rss_bytes").process_rss_bytes

    def static_facts(self) -> HostFacts:
        """Return the next scripted static host facts."""
        return self._next("static_facts").static_facts


class ScriptedGpuReader:
    """Replay fixed per-device GPU sample sets in deterministic order.

    ``sample()`` consumes one outer sequence element. ``available()`` reports whether that next
    element contains a GPU without consuming it, and ``static_info()`` derives stable profiles from
    the first occurrence of each device. Exhaustion raises ``IndexError`` rather than repeating a
    stale sample. Access is serialized and safe across threads.

    Args:
        samples: Per-tick GPU sample sequences, copied recursively during construction.
    """

    def __init__(self, samples: Sequence[Sequence[GpuSample]]) -> None:
        """Copy the script and initialize its live-sample cursor."""
        self._samples = tuple(tuple(sample) for sample in samples)
        self._position = 0
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Return whether the next unconsumed script step contains at least one GPU."""
        with self._lock:
            return self._position < len(self._samples) and bool(self._samples[self._position])

    def sample(self) -> tuple[GpuSample, ...]:
        """Return and consume the next scripted per-device GPU sample set."""
        with self._lock:
            position = self._position
            if position >= len(self._samples):
                raise IndexError(f"ScriptedGpuReader.sample exhausted at step {position}.")
            self._position = position + 1
            return self._samples[position]

    def static_info(self) -> tuple[GpuProfile, ...]:
        """Derive one static profile from the first scripted occurrence of each GPU index."""
        profiles: dict[int, GpuProfile] = {}
        for samples in self._samples:
            for sample in samples:
                profiles.setdefault(
                    sample.index,
                    GpuProfile(
                        index=sample.index,
                        name=None,
                        uuid=sample.uuid,
                        vram_total_bytes=sample.vram_total_bytes,
                        vendor=GpuVendor.UNKNOWN,
                    ),
                )
        return tuple(profiles[index] for index in sorted(profiles))


class FaultInjectingReader:
    """Wrap a host or GPU reader and raise at exactly one named reader operation.

    The wrapper implements both protocols so it can be passed directly to ``TelemetryCollector``;
    operations outside the wrapped protocol raise an actionable ``ValidationError``. The injected
    ``RuntimeError`` is deterministic and leaves every other operation delegated unchanged.

    Args:
        wrapped: Host or GPU reader to delegate to.
        fail: Exact public operation name at which to raise.

    Raises:
        ValidationError: If ``wrapped`` implements neither reader protocol or ``fail`` is not an
            operation on the implemented protocol.
    """

    def __init__(self, wrapped: HostReader | GpuReader, *, fail: str) -> None:
        """Validate the target protocol and configure one injected failure point."""
        is_host = isinstance(wrapped, HostReader)
        is_gpu = isinstance(wrapped, GpuReader)
        allowed = (_HOST_OPERATIONS if is_host else frozenset()) | (
            _GPU_OPERATIONS if is_gpu else frozenset()
        )
        if not allowed:
            raise ValidationError(
                "FaultInjectingReader.wrapped must implement HostReader or GpuReader.",
                details={"field": "wrapped"},
            )
        if fail not in allowed:
            raise ValidationError(
                f"FaultInjectingReader.fail must name one of {sorted(allowed)!r}; got {fail!r}.",
                details={"field": "fail", "value": fail},
            )
        self._wrapped = wrapped
        self._is_host = is_host
        self._is_gpu = is_gpu
        self._fail = fail

    def _call[T](self, operation: str, fn: Callable[[], T]) -> T:
        """Raise at the configured operation or delegate unchanged."""
        if operation == self._fail:
            raise RuntimeError(f"Injected telemetry fault at {operation}.")
        return fn()

    def _host(self) -> HostReader:
        """Return the wrapped host view or reject a protocol-mismatched call."""
        if not self._is_host:
            raise ValidationError(
                "FaultInjectingReader wraps a GPU reader; host operation requested.",
                details={"operation_kind": "host"},
            )
        return cast(HostReader, self._wrapped)

    def _gpu(self) -> GpuReader:
        """Return the wrapped GPU view or reject a protocol-mismatched call."""
        if not self._is_gpu:
            raise ValidationError(
                "FaultInjectingReader wraps a host reader; GPU operation requested.",
                details={"operation_kind": "gpu"},
            )
        return cast(GpuReader, self._wrapped)

    def cpu_percent(self) -> Measurement:
        """Delegate or inject the configured CPU operation fault."""
        return self._call("cpu_percent", self._host().cpu_percent)

    def load_average_1m(self) -> Measurement:
        """Delegate or inject the configured load operation fault."""
        return self._call("load_average_1m", self._host().load_average_1m)

    def memory(self) -> MemoryReading:
        """Delegate or inject the configured memory operation fault."""
        return self._call("memory", self._host().memory)

    def cpu_temperature(self) -> Measurement:
        """Delegate or inject the configured temperature operation fault."""
        return self._call("cpu_temperature", self._host().cpu_temperature)

    def disk_throughput(self) -> DiskThroughput:
        """Delegate or inject the configured disk operation fault."""
        return self._call("disk_throughput", self._host().disk_throughput)

    def process_rss_bytes(self) -> Measurement:
        """Delegate or inject the configured process-memory operation fault."""
        return self._call("process_rss_bytes", self._host().process_rss_bytes)

    def static_facts(self) -> HostFacts:
        """Delegate or inject the configured static-host operation fault."""
        return self._call("static_facts", self._host().static_facts)

    def available(self) -> bool:
        """Delegate or inject the configured GPU availability fault."""
        return self._call("available", self._gpu().available)

    def sample(self) -> Sequence[GpuSample]:
        """Delegate or inject the configured live-GPU operation fault."""
        return self._call("sample", self._gpu().sample)

    def static_info(self) -> Sequence[GpuProfile]:
        """Delegate or inject the configured static-GPU operation fault."""
        return self._call("static_info", self._gpu().static_info)

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Delegate optional wrapped-reader diagnostics without altering them."""
        reporter = getattr(self._wrapped, "unavailable_reasons", None)
        if not callable(reporter):
            return {}
        return cast(Mapping[str, str], reporter())
