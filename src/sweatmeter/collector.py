"""Non-raising orchestration of live readers and static machine profiling."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from baseaicore import (
    UNSUPPORTED,
    Clock,
    GpuProfile,
    GpuVendor,
    MachineProfile,
    Measurement,
    Unsupported,
    compute_machine_fingerprint,
    is_supported,
    utc_now,
)

from sweatmeter.platform import create_gpu_reader, create_host_reader
from sweatmeter.readers.protocols import GpuReader, HostReader
from sweatmeter.safe import _safe
from sweatmeter.types import DiskThroughput, GpuSample, HostFacts, MemoryReading, TelemetrySnapshot

__all__ = ["TelemetryCollector"]

_FALLBACK_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)
_GPU_MEASUREMENT_FIELDS = (
    "utilization_percent",
    "memory_utilization_percent",
    "vram_used_bytes",
    "vram_total_bytes",
    "temperature_c",
    "memory_temperature_c",
    "power_watts",
    "power_limit_watts",
    "fan_percent",
    "core_clock_mhz",
    "memory_clock_mhz",
)


class _ReadFailure:
    """Mark an exception separately from a legitimate unsupported result."""


_READ_FAILURE = _ReadFailure()


@runtime_checkable
class _ReasonedReader(Protocol):
    """Describe the optional diagnostics surface implemented by concrete readers."""

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return reasons from the reader's most recent operation."""
        ...


def _number(
    value: object, *, minimum: float | None = 0.0, maximum: float | None = None
) -> Measurement:
    """Return a finite in-range measurement or the unsupported sentinel."""
    if value is UNSUPPORTED:
        return UNSUPPORTED
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNSUPPORTED
    if not math.isfinite(value):
        return UNSUPPORTED
    if minimum is not None and value < minimum:
        return UNSUPPORTED
    if maximum is not None and value > maximum:
        return UNSUPPORTED
    return value


def _profile_text(value: object) -> str | None:
    """Normalize unavailable or blank profile text to BaseAiCore's ``None`` form."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _fallback_timestamp() -> datetime:
    """Return a real UTC instant, with an epoch last resort for clock failure."""
    value = _safe(utc_now, _FALLBACK_INSTANT)
    if not isinstance(value, datetime):
        return _FALLBACK_INSTANT
    converted = _safe(lambda: value.astimezone(UTC), _FALLBACK_INSTANT)
    return converted if isinstance(converted, datetime) else _FALLBACK_INSTANT


class TelemetryCollector:
    """Compose platform readers into snapshots and stable machine profiles.

    Ordinary reader, clock, parser, and validation failures never escape ``snapshot()`` or
    ``machine_profile()``. Each boundary call is isolated, unavailable measurements remain
    ``UNSUPPORTED``, and diagnostics name the affected field. Calls are serialized because the
    Linux delta readers and NVIDIA diagnostic map are stateful; the collector is therefore safe to
    share with one background sampler and synchronous consumers.

    Args:
        host: Host reader. Defaults to the current platform factory.
        gpu: GPU reader. Defaults to the configured NVIDIA command backend.
        clock: Injectable wall clock returning a timezone-aware instant.

    Raises:
        UnsupportedPlatformError: During construction only, if no default host implementation or
            tier-3 stub exists. Pass ``NullHostReader`` to degrade explicitly on such a platform.
    """

    def __init__(
        self,
        *,
        host: HostReader | None = None,
        gpu: GpuReader | None = None,
        clock: Clock = utc_now,
    ) -> None:
        """Install injectable readers and the timestamp source."""
        self._host = create_host_reader() if host is None else host
        self._gpu = create_gpu_reader() if gpu is None else gpu
        self._clock = clock
        self._reasons: dict[str, str] = {}
        self._lock = threading.RLock()

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return a copy of diagnostics from the most recent collection operation."""
        with self._lock:
            return dict(self._reasons)

    def snapshot(self) -> TelemetrySnapshot:
        """Collect one complete live snapshot without propagating ordinary failures.

        Returns:
            A timestamped observation. Any failed or malformed field is ``UNSUPPORTED`` and has an
            entry in both the snapshot's and collector's diagnostic mapping.
        """
        with self._lock:
            collected = _safe(self._snapshot, _READ_FAILURE)
            snapshot = (
                collected
                if isinstance(collected, TelemetrySnapshot)
                else TelemetrySnapshot(
                    timestamp=_fallback_timestamp(),
                    _reasons=(("snapshot", "collector_error"),),
                )
            )
            self._reasons = dict(snapshot.unavailable_reasons())
            return snapshot

    def machine_profile(self) -> MachineProfile:
        """Collect static facts and compute their stable BaseAiCore fingerprint without raising.

        GPU profiles are sorted by device index before storage. Driver, toolkit, OS/kernel,
        storage, Python version, and observation time remain recorded provenance but are excluded
        by BaseAiCore's fingerprint function.

        Returns:
            One immutable profile carrying its fingerprint and observation time. Failed fields use
            BaseAiCore's explicit unsupported representation and are recorded in diagnostics. An
            internal assembly failure yields the all-unsupported profile and the single diagnostic
            ``machine_profile: collector_error``, so a caller can tell a wholly failed collection
            from a machine that merely reports little about itself.
        """
        with self._lock:
            profile = _safe(self._machine_profile, _READ_FAILURE)
            if not isinstance(profile, MachineProfile):
                self._reasons = {"machine_profile": "collector_error"}
                return self._emergency_profile()
            return profile

    def _read[T, D](
        self,
        fn: Callable[[], T],
        default: D,
        *,
        reason_fields: Sequence[str],
        reasons: dict[str, str],
    ) -> T | D:
        """Read one boundary and attach a reason only when that call raises."""
        value = _safe(fn, _READ_FAILURE)
        if isinstance(value, _ReadFailure):
            for field in reason_fields:
                reasons[field] = "reader_error"
            return default
        return value

    def _timestamp(self, reasons: dict[str, str]) -> datetime:
        """Read and normalize the injected timestamp, falling back safely when invalid."""
        raw = self._read(
            self._clock,
            _FALLBACK_INSTANT,
            reason_fields=("timestamp",),
            reasons=reasons,
        )
        if isinstance(raw, datetime):
            timezone = raw.tzinfo
            if timezone is not None and _safe(lambda: timezone.utcoffset(raw), None) is not None:
                converted = _safe(lambda: raw.astimezone(UTC), None)
                if isinstance(converted, datetime):
                    return converted
        reasons["timestamp"] = "invalid_clock"
        return _fallback_timestamp()

    @staticmethod
    def _reader_reasons(reader: object) -> dict[str, str]:
        """Copy well-formed diagnostics from an optional reader reason surface."""
        if not isinstance(reader, _ReasonedReader):
            return {}
        reported = _safe(reader.unavailable_reasons, {})
        if not isinstance(reported, Mapping):
            return {}
        normalized: dict[str, str] = {}
        items = _safe(lambda: tuple(reported.items()), ())
        for key, value in items:
            if isinstance(key, str) and key and isinstance(value, str) and value:
                normalized[key] = value
        return normalized

    @staticmethod
    def _mark_unsupported(
        reasons: dict[str, str], values: Mapping[str, Measurement], *, reason: str
    ) -> None:
        """Ensure every unsupported measurement has a diagnostic explanation."""
        for field, value in values.items():
            if not is_supported(value):
                reasons.setdefault(field, reason)

    def _snapshot(self) -> TelemetrySnapshot:
        """Collect field-isolated live values under the public non-raising wrapper."""
        reasons: dict[str, str] = {}
        timestamp = self._timestamp(reasons)
        cpu = _number(
            self._read(
                self._host.cpu_percent,
                UNSUPPORTED,
                reason_fields=("cpu_percent",),
                reasons=reasons,
            ),
            maximum=100.0,
        )
        load = _number(
            self._read(
                self._host.load_average_1m,
                UNSUPPORTED,
                reason_fields=("load_average_1m",),
                reasons=reasons,
            )
        )
        memory = self._read(
            self._host.memory,
            MemoryReading(),
            reason_fields=("ram_used_bytes", "ram_available_bytes", "ram_total_bytes"),
            reasons=reasons,
        )
        if not isinstance(memory, MemoryReading):
            memory = MemoryReading()
            for field in ("ram_used_bytes", "ram_available_bytes", "ram_total_bytes"):
                reasons[field] = "malformed_reading"
        temperature = _number(
            self._read(
                self._host.cpu_temperature,
                UNSUPPORTED,
                reason_fields=("cpu_temperature_c",),
                reasons=reasons,
            ),
            minimum=-273.15,
        )
        disk = self._read(
            self._host.disk_throughput,
            DiskThroughput(),
            reason_fields=("disk_read_bytes_per_sec", "disk_write_bytes_per_sec"),
            reasons=reasons,
        )
        if not isinstance(disk, DiskThroughput):
            disk = DiskThroughput()
            reasons["disk_read_bytes_per_sec"] = "malformed_reading"
            reasons["disk_write_bytes_per_sec"] = "malformed_reading"
        process_rss = _number(
            self._read(
                self._host.process_rss_bytes,
                UNSUPPORTED,
                reason_fields=("process_rss_bytes",),
                reasons=reasons,
            )
        )

        gpu_values = self._read(
            lambda: tuple(self._gpu.sample()),
            (),
            reason_fields=("gpu",),
            reasons=reasons,
        )
        gpus = self._normalize_gpu_samples(gpu_values, reasons)

        live_fields = {
            "cpu_percent": cpu,
            "load_average_1m": load,
            "ram_used_bytes": _number(memory.used_bytes),
            "ram_available_bytes": _number(memory.available_bytes),
            "ram_total_bytes": _number(memory.total_bytes),
            "cpu_temperature_c": temperature,
            "disk_read_bytes_per_sec": _number(disk.read_bytes_per_sec),
            "disk_write_bytes_per_sec": _number(disk.write_bytes_per_sec),
            "process_rss_bytes": process_rss,
        }
        self._mark_unsupported(reasons, live_fields, reason="sensor_unsupported")
        for key, value in self._reader_reasons(self._host).items():
            if key in live_fields:
                reasons[key] = value
        reasons.update(self._reader_reasons(self._gpu))
        if not gpus:
            reasons.setdefault("gpu", "no_gpus")

        return TelemetrySnapshot(
            timestamp=timestamp,
            cpu_percent=live_fields["cpu_percent"],
            load_average_1m=live_fields["load_average_1m"],
            ram_used_bytes=live_fields["ram_used_bytes"],
            ram_available_bytes=live_fields["ram_available_bytes"],
            ram_total_bytes=live_fields["ram_total_bytes"],
            cpu_temperature_c=live_fields["cpu_temperature_c"],
            disk_read_bytes_per_sec=live_fields["disk_read_bytes_per_sec"],
            disk_write_bytes_per_sec=live_fields["disk_write_bytes_per_sec"],
            process_rss_bytes=live_fields["process_rss_bytes"],
            gpus=gpus,
            _reasons=tuple(sorted(reasons.items())),
        )

    @staticmethod
    def _normalize_gpu_samples(
        values: tuple[object, ...], reasons: dict[str, str]
    ) -> tuple[GpuSample, ...]:
        """Retain valid GPU rows, normalize their fields, and sort by device index."""
        normalized: dict[int, GpuSample] = {}
        for position, value in enumerate(values):
            if not isinstance(value, GpuSample):
                reasons[f"gpu.sample.{position}"] = "malformed_reading"
                continue
            index = value.index
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                reasons[f"gpu.sample.{position}.index"] = "malformed_value"
                continue
            if index in normalized:
                reasons[f"gpu.sample.{position}.index"] = "duplicate_gpu_index"
                continue
            measurements: dict[str, Measurement] = {}
            for field in _GPU_MEASUREMENT_FIELDS:
                maximum = 100.0 if field.endswith("percent") else None
                minimum = -273.15 if "temperature" in field else 0.0
                measurement = _number(getattr(value, field), minimum=minimum, maximum=maximum)
                measurements[field] = measurement
                if not is_supported(measurement):
                    reasons.setdefault(f"gpu.{index}.{field}", "sensor_unsupported")
            normalized[index] = GpuSample(
                index=index,
                uuid=value.uuid if isinstance(value.uuid, str) and value.uuid else None,
                utilization_percent=measurements["utilization_percent"],
                memory_utilization_percent=measurements["memory_utilization_percent"],
                vram_used_bytes=measurements["vram_used_bytes"],
                vram_total_bytes=measurements["vram_total_bytes"],
                temperature_c=measurements["temperature_c"],
                memory_temperature_c=measurements["memory_temperature_c"],
                power_watts=measurements["power_watts"],
                power_limit_watts=measurements["power_limit_watts"],
                fan_percent=measurements["fan_percent"],
                core_clock_mhz=measurements["core_clock_mhz"],
                memory_clock_mhz=measurements["memory_clock_mhz"],
                throttle_reasons=tuple(
                    reason
                    for reason in value.throttle_reasons
                    if isinstance(reason, str) and reason
                ),
                throttle_reasons_available=value.throttle_reasons_available is True,
            )
        return tuple(normalized[index] for index in sorted(normalized))

    def _machine_profile(self) -> MachineProfile:
        """Assemble a normalized profile under the public non-raising wrapper."""
        reasons: dict[str, str] = {}
        facts = self._read(
            self._host.static_facts,
            HostFacts(),
            reason_fields=(
                "hostname",
                "os_name",
                "os_version",
                "kernel",
                "architecture",
                "cpu_model",
                "physical_cores",
                "logical_cores",
                "ram_bytes",
                "storage",
                "python_version",
            ),
            reasons=reasons,
        )
        if not isinstance(facts, HostFacts):
            facts = HostFacts()
            reasons["host_profile"] = "malformed_reading"
        gpu_values = self._read(
            lambda: tuple(self._gpu.static_info()),
            (),
            reason_fields=("gpu",),
            reasons=reasons,
        )
        gpus = self._normalize_gpu_profiles(gpu_values, reasons)
        timestamp = self._timestamp(reasons)

        hostname = _profile_text(facts.hostname)
        os_name = _profile_text(facts.os_name)
        architecture = _profile_text(facts.architecture)
        cpu_model = _profile_text(facts.cpu_model)
        physical_cores = _number(facts.physical_cores)
        logical_cores = _number(facts.logical_cores)
        ram_bytes = _number(facts.ram_bytes)
        fingerprint = compute_machine_fingerprint(
            hostname=hostname,
            os_name=os_name,
            architecture=architecture,
            cpu_model=cpu_model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            ram_bytes=ram_bytes,
            gpus=gpus,
        )

        profile = MachineProfile(
            machine_fingerprint=fingerprint,
            hostname=hostname,
            os_name=os_name,
            os_version=_profile_text(facts.os_version),
            kernel=_profile_text(facts.kernel),
            architecture=architecture,
            cpu_model=cpu_model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            ram_bytes=ram_bytes,
            gpus=gpus,
            storage=facts.storage if isinstance(facts.storage, tuple) else (),
            python_version=_profile_text(facts.python_version),
            observed_at=timestamp,
        )
        identity_values: dict[str, Measurement | Unsupported] = {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "ram_bytes": ram_bytes,
        }
        self._mark_unsupported(reasons, identity_values, reason="sensor_unsupported")
        for field, value in (
            ("hostname", hostname),
            ("os_name", os_name),
            ("architecture", architecture),
            ("cpu_model", cpu_model),
        ):
            if value is None:
                reasons.setdefault(field, "sensor_unsupported")
        for key, value in self._reader_reasons(self._host).items():
            if key not in {
                "cpu_percent",
                "load_average_1m",
                "ram_used_bytes",
                "ram_available_bytes",
                "ram_total_bytes",
                "cpu_temperature_c",
                "disk_read_bytes_per_sec",
                "disk_write_bytes_per_sec",
                "process_rss_bytes",
            }:
                reasons[key] = value
        reasons.update(self._reader_reasons(self._gpu))
        if not gpus:
            reasons.setdefault("gpu", "no_gpus")
        self._reasons = dict(sorted(reasons.items()))
        return profile

    @staticmethod
    def _normalize_gpu_profiles(
        values: tuple[object, ...], reasons: dict[str, str]
    ) -> tuple[GpuProfile, ...]:
        """Retain valid static GPU profiles and normalize their sequence by index."""
        normalized: dict[int, GpuProfile] = {}
        for position, value in enumerate(values):
            if not isinstance(value, GpuProfile):
                reasons[f"gpu.profile.{position}"] = "malformed_reading"
                continue
            if value.index in normalized:
                reasons[f"gpu.profile.{position}.index"] = "duplicate_gpu_index"
                continue
            profile = GpuProfile(
                index=value.index,
                name=_profile_text(value.name),
                uuid=_profile_text(value.uuid),
                vram_total_bytes=_number(value.vram_total_bytes),
                driver_version=_profile_text(value.driver_version),
                cuda_version=_profile_text(value.cuda_version),
                compute_capability=_profile_text(value.compute_capability),
                vendor=value.vendor if isinstance(value.vendor, GpuVendor) else GpuVendor.UNKNOWN,
            )
            normalized[profile.index] = profile
        return tuple(normalized[index] for index in sorted(normalized))

    @staticmethod
    def _emergency_profile() -> MachineProfile:
        """Build the stable all-unsupported profile used after an internal collector failure."""
        fingerprint = compute_machine_fingerprint(
            hostname=None,
            os_name=None,
            architecture=None,
            cpu_model=None,
            physical_cores=UNSUPPORTED,
            logical_cores=UNSUPPORTED,
            ram_bytes=UNSUPPORTED,
            gpus=(),
        )
        return MachineProfile(
            machine_fingerprint=fingerprint,
            hostname=None,
            os_name=None,
            os_version=None,
            kernel=None,
            architecture=None,
            cpu_model=None,
            observed_at=_fallback_timestamp(),
        )
