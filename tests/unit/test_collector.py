"""Snapshot, machine-profile, diagnostic, and supported-double tests for Phase 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pytest
from baseaicore import (
    UNSUPPORTED,
    GpuProfile,
    GpuVendor,
    MachineProfile,
    StorageDevice,
    ValidationError,
    compute_machine_fingerprint,
)

from sweatmeter import (
    GpuReader,
    GpuSample,
    HostReader,
    NullGpuReader,
    NullHostReader,
    TelemetryCollector,
    TelemetrySnapshot,
)
from sweatmeter.testing import (
    FaultInjectingReader,
    HostReading,
    ScriptedGpuReader,
    ScriptedHostReader,
)
from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading

_NOW = datetime(2026, 8, 22, 18, 30, tzinfo=UTC)


def _gpu_sample(*, index: int = 0, uuid: str = "GPU-0") -> GpuSample:
    return GpuSample(
        index=index,
        uuid=uuid,
        utilization_percent=72.5,
        memory_utilization_percent=44.0,
        vram_used_bytes=4_000,
        vram_total_bytes=16_000,
        temperature_c=68.0,
        memory_temperature_c=72.0,
        power_watts=115.5,
        power_limit_watts=180.0,
        fan_percent=40.0,
        core_clock_mhz=2_400.0,
        memory_clock_mhz=9_000.0,
        throttle_reasons=(),
        throttle_reasons_available=True,
    )


def _host_reading(*, cpu_percent: float = 25.0) -> HostReading:
    return HostReading(
        cpu_percent=cpu_percent,
        load_average_1m=1.25,
        memory=MemoryReading(total_bytes=32_000, available_bytes=20_000, used_bytes=12_000),
        cpu_temperature_c=55.0,
        disk_throughput=DiskThroughput(read_bytes_per_sec=1_500, write_bytes_per_sec=750),
        process_rss_bytes=12_345,
        static_facts=HostFacts(
            hostname="workstation",
            os_name="Linux",
            os_version="Test OS 1",
            kernel="6.14-test",
            architecture="x86_64",
            cpu_model="Test CPU",
            physical_cores=8,
            logical_cores=16,
            ram_bytes=32_000,
            storage=(StorageDevice("nvme0n1", 1_000_000, "Test Disk", False),),
            python_version="3.13.15",
        ),
    )


class StaticGpuReader:
    def __init__(self, profiles: Sequence[GpuProfile]) -> None:
        self._profiles = tuple(profiles)

    def available(self) -> bool:
        return bool(self._profiles)

    def sample(self) -> tuple[GpuSample, ...]:
        return ()

    def static_info(self) -> tuple[GpuProfile, ...]:
        return self._profiles


class InternallyBrokenCollector(TelemetryCollector):
    def _snapshot(self) -> TelemetrySnapshot:
        raise RuntimeError("internal snapshot defect")

    def _machine_profile(self) -> MachineProfile:
        raise RuntimeError("internal profile defect")


def _naive_clock() -> datetime:
    return _NOW.replace(tzinfo=None)


def _raising_clock() -> datetime:
    raise OSError("clock failed")


def test_snapshot_composes_complete_host_and_gpu_reading() -> None:
    collector = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]),
        gpu=ScriptedGpuReader([[_gpu_sample()]]),
        clock=lambda: _NOW,
    )

    snapshot = collector.snapshot()

    assert snapshot.timestamp == _NOW
    assert snapshot.cpu_percent == 25.0
    assert snapshot.load_average_1m == 1.25
    assert snapshot.ram_used_bytes == 12_000
    assert snapshot.ram_available_bytes == 20_000
    assert snapshot.ram_total_bytes == 32_000
    assert snapshot.cpu_temperature_c == 55.0
    assert snapshot.disk_read_bytes_per_sec == 1_500
    assert snapshot.disk_write_bytes_per_sec == 750
    assert snapshot.process_rss_bytes == 12_345
    assert snapshot.gpus == (_gpu_sample(),)
    assert snapshot.unavailable_reasons() == {}
    assert collector.unavailable_reasons() == {}


@pytest.mark.parametrize(
    ("operation", "degraded_fields"),
    [
        ("cpu_percent", {"cpu_percent"}),
        ("load_average_1m", {"load_average_1m"}),
        ("memory", {"ram_used_bytes", "ram_available_bytes", "ram_total_bytes"}),
        ("cpu_temperature", {"cpu_temperature_c"}),
        ("disk_throughput", {"disk_read_bytes_per_sec", "disk_write_bytes_per_sec"}),
        ("process_rss_bytes", {"process_rss_bytes"}),
    ],
)
def test_snapshot_isolates_every_host_reader_failure(
    operation: str, degraded_fields: set[str]
) -> None:
    host = FaultInjectingReader(ScriptedHostReader([_host_reading()]), fail=operation)
    collector = TelemetryCollector(
        host=host,
        gpu=ScriptedGpuReader([[_gpu_sample()]]),
        clock=lambda: _NOW,
    )

    snapshot = collector.snapshot()
    reasons = snapshot.unavailable_reasons()

    assert {
        field for field, reason in reasons.items() if reason == "reader_error"
    } == degraded_fields
    assert snapshot.gpus == (_gpu_sample(),)
    if "cpu_percent" not in degraded_fields:
        assert snapshot.cpu_percent == 25.0
    if "ram_total_bytes" not in degraded_fields:
        assert snapshot.ram_total_bytes == 32_000


def test_snapshot_isolates_gpu_reader_failure_without_degrading_host() -> None:
    gpu = FaultInjectingReader(ScriptedGpuReader([[_gpu_sample()]]), fail="sample")
    collector = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]), gpu=gpu, clock=lambda: _NOW
    )

    snapshot = collector.snapshot()

    assert snapshot.cpu_percent == 25.0
    assert snapshot.gpus == ()
    assert snapshot.unavailable_reasons()["gpu"] == "reader_error"


def test_snapshot_normalizes_gpu_order_and_rejects_duplicate_index() -> None:
    gpu = ScriptedGpuReader([[_gpu_sample(index=2, uuid="GPU-2"), _gpu_sample()]])
    collector = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]), gpu=gpu, clock=lambda: _NOW
    )

    snapshot = collector.snapshot()

    assert [sample.index for sample in snapshot.gpus] == [0, 2]


def test_snapshot_records_reasons_for_every_null_host_field() -> None:
    collector = TelemetryCollector(host=NullHostReader(), gpu=NullGpuReader(), clock=lambda: _NOW)

    snapshot = collector.snapshot()
    reasons = snapshot.unavailable_reasons()

    for field in (
        "cpu_percent",
        "load_average_1m",
        "ram_used_bytes",
        "ram_available_bytes",
        "ram_total_bytes",
        "cpu_temperature_c",
        "disk_read_bytes_per_sec",
        "disk_write_bytes_per_sec",
        "process_rss_bytes",
    ):
        assert getattr(snapshot, field) is UNSUPPORTED
        assert reasons[field] == "platform_unsupported"
    assert reasons["gpu"] == "platform_unsupported"


def test_snapshot_reason_mapping_is_a_defensive_copy() -> None:
    collector = TelemetryCollector(host=NullHostReader(), gpu=NullGpuReader(), clock=lambda: _NOW)
    snapshot = collector.snapshot()
    reasons = dict(snapshot.unavailable_reasons())

    reasons["cpu_percent"] = "tampered"

    assert snapshot.unavailable_reasons()["cpu_percent"] == "platform_unsupported"
    assert collector.unavailable_reasons()["cpu_percent"] == "platform_unsupported"


@pytest.mark.parametrize("clock", [_naive_clock, _raising_clock])
def test_snapshot_bad_clock_falls_back_to_aware_utc(clock: Callable[[], datetime]) -> None:
    collector = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]),
        gpu=ScriptedGpuReader([[_gpu_sample()]]),
        clock=clock,
    )

    snapshot = collector.snapshot()

    assert snapshot.timestamp.tzinfo is not None
    assert snapshot.timestamp.utcoffset() is not None
    assert snapshot.unavailable_reasons()["timestamp"] in {"invalid_clock", "reader_error"}


def test_machine_profile_assembles_static_facts_and_sorts_gpus() -> None:
    profiles = (
        GpuProfile(2, "GPU C", "UUID-C", 24_000, "580", "13.0", "12.0", GpuVendor.NVIDIA),
        GpuProfile(0, "GPU A", "UUID-A", 16_000, "580", "13.0", "12.0", GpuVendor.NVIDIA),
    )
    collector = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]),
        gpu=StaticGpuReader(profiles),
        clock=lambda: _NOW,
    )

    profile = collector.machine_profile()

    assert profile.hostname == "workstation"
    assert profile.cpu_model == "Test CPU"
    assert profile.ram_bytes == 32_000
    assert [gpu.index for gpu in profile.gpus] == [0, 2]
    assert profile.observed_at == _NOW
    assert profile.machine_fingerprint == compute_machine_fingerprint(
        hostname="workstation",
        os_name="Linux",
        architecture="x86_64",
        cpu_model="Test CPU",
        physical_cores=8,
        logical_cores=16,
        ram_bytes=32_000,
        gpus=profile.gpus,
    )


def test_machine_fingerprint_excludes_driver_versions_and_input_order() -> None:
    first = (
        GpuProfile(1, "GPU B", "UUID-B", driver_version="old"),
        GpuProfile(0, "GPU A", "UUID-A", driver_version="old"),
    )
    second = (
        GpuProfile(0, "GPU A", "UUID-A", driver_version="new"),
        GpuProfile(1, "GPU B", "UUID-B", driver_version="new"),
    )
    first_profile = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]),
        gpu=StaticGpuReader(first),
        clock=lambda: _NOW,
    ).machine_profile()
    second_profile = TelemetryCollector(
        host=ScriptedHostReader([_host_reading()]),
        gpu=StaticGpuReader(second),
        clock=lambda: _NOW,
    ).machine_profile()

    assert first_profile.machine_fingerprint == second_profile.machine_fingerprint
    assert [gpu.driver_version for gpu in first_profile.gpus] == ["old", "old"]
    assert [gpu.driver_version for gpu in second_profile.gpus] == ["new", "new"]


@pytest.mark.parametrize("kind", ["host", "gpu"])
def test_machine_profile_isolates_static_reader_failures(kind: str) -> None:
    host = ScriptedHostReader([_host_reading()])
    gpu = ScriptedGpuReader([[_gpu_sample()]])
    host_reader: HostReader
    gpu_reader: GpuReader
    if kind == "host":
        host_reader = FaultInjectingReader(host, fail="static_facts")
        gpu_reader = gpu
    else:
        host_reader = host
        gpu_reader = FaultInjectingReader(gpu, fail="static_info")
    collector = TelemetryCollector(host=host_reader, gpu=gpu_reader, clock=lambda: _NOW)

    profile = collector.machine_profile()

    assert len(profile.machine_fingerprint) == 64
    assert collector.unavailable_reasons()["gpu" if kind == "gpu" else "hostname"] == "reader_error"


def test_public_methods_have_emergency_fallback_for_internal_defects() -> None:
    collector = InternallyBrokenCollector(
        host=NullHostReader(), gpu=NullGpuReader(), clock=lambda: _NOW
    )

    snapshot = collector.snapshot()
    profile = collector.machine_profile()

    assert snapshot.unavailable_reasons() == {"snapshot": "collector_error"}
    assert len(profile.machine_fingerprint) == 64


def test_telemetry_snapshot_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        TelemetrySnapshot(_NOW.replace(tzinfo=None))


def test_scripted_host_reader_uses_independent_deterministic_cursors() -> None:
    reader = ScriptedHostReader([_host_reading(cpu_percent=10), _host_reading(cpu_percent=20)])

    assert reader.cpu_percent() == 10
    assert reader.cpu_percent() == 20
    assert reader.memory().total_bytes == 32_000
    assert reader.memory().total_bytes == 32_000
    with pytest.raises(IndexError, match="cpu_percent exhausted"):
        reader.cpu_percent()


def test_scripted_gpu_reader_replays_and_derives_static_profiles() -> None:
    reader = ScriptedGpuReader([[_gpu_sample()], [_gpu_sample(index=1, uuid="GPU-1")]])

    assert reader.available() is True
    assert reader.sample() == (_gpu_sample(),)
    assert reader.sample() == (_gpu_sample(index=1, uuid="GPU-1"),)
    assert [profile.index for profile in reader.static_info()] == [0, 1]
    assert reader.available() is False
    with pytest.raises(IndexError, match="sample exhausted"):
        reader.sample()


def test_fault_injecting_reader_fails_only_named_operation() -> None:
    reader = FaultInjectingReader(ScriptedHostReader([_host_reading()]), fail="cpu_percent")

    with pytest.raises(RuntimeError, match="cpu_percent"):
        reader.cpu_percent()
    assert reader.memory().total_bytes == 32_000


def test_fault_injecting_reader_rejects_unknown_operation() -> None:
    with pytest.raises(ValidationError, match="must name one of"):
        FaultInjectingReader(ScriptedHostReader([_host_reading()]), fail="not_a_field")
