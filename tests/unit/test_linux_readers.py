"""Fixture-driven tests for the Phase 1 Linux host readers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, get_type_hints

import pytest
from baseaicore import UNSUPPORTED, Measurement, Unsupported

from sweatmeter.readers import (
    HostReader,
    LinuxHostReader,
    parse_cpuinfo,
    parse_diskstats,
    parse_loadavg,
    parse_meminfo,
    parse_proc_stat,
    read_block_devices,
    read_hwmon,
    read_thermal_zones,
)
from sweatmeter.readers.linux import _reported_text
from sweatmeter.safe import _safe
from sweatmeter.types import MemoryReading, ReportedText

if TYPE_CHECKING:
    from collections.abc import Callable


def _fixture(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def test_parse_proc_stat_uses_aggregate_and_does_not_double_count_guest(
    telemetry_fixtures: Path,
) -> None:
    parsed = parse_proc_stat(_fixture(telemetry_fixtures, "proc/stat-first.txt"))

    assert parsed == (1000, 800)


@pytest.mark.parametrize(
    "text",
    ["", "cpu0 1 2 3 4\n", "cpu 1 2 3\n", "cpu 1 two 3 4\n", "cpu 1 2 3 -4\n"],
)
def test_parse_proc_stat_malformed_is_unsupported(text: str) -> None:
    assert parse_proc_stat(text) is UNSUPPORTED


def test_cpu_percent_first_call_is_unsupported_then_uses_counter_delta(
    telemetry_fixtures: Path,
) -> None:
    samples = iter(
        [
            _fixture(telemetry_fixtures, "proc/stat-first.txt"),
            _fixture(telemetry_fixtures, "proc/stat-second.txt"),
        ]
    )
    reader = LinuxHostReader(read_text=lambda _path: next(samples))

    first = reader.cpu_percent()
    second = reader.cpu_percent()

    assert first is UNSUPPORTED
    assert second == pytest.approx(200 / 300 * 100)


def test_cpu_percent_counter_rollback_is_unsupported_and_resets_baseline() -> None:
    samples = iter(
        [
            "cpu 100 0 100 800\n",
            "cpu 10 0 10 80\n",
            "cpu 20 0 20 90\n",
        ]
    )
    reader = LinuxHostReader(read_text=lambda _path: next(samples))

    assert reader.cpu_percent() is UNSUPPORTED
    assert reader.cpu_percent() is UNSUPPORTED
    assert reader.cpu_percent() == pytest.approx(200 / 300 * 100)


def test_cpu_percent_read_failure_does_not_discard_last_valid_baseline() -> None:
    calls = 0

    def read_text(_path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("fixture denied")
        return "cpu 100 0 100 800\n" if calls == 1 else "cpu 200 0 200 900\n"

    reader = LinuxHostReader(read_text=read_text)

    assert reader.cpu_percent() is UNSUPPORTED
    assert reader.cpu_percent() is UNSUPPORTED
    assert reader.cpu_percent() == pytest.approx(200 / 300 * 100)


def test_parse_meminfo_normalizes_bytes_and_derives_used(telemetry_fixtures: Path) -> None:
    reading = parse_meminfo(_fixture(telemetry_fixtures, "proc/meminfo-normal.txt"))

    assert reading.total_bytes == 16_384_000 * 1024
    assert reading.available_bytes == 4_096_000 * 1024
    assert reading.used_bytes == 12_288_000 * 1024


def test_parse_meminfo_minimal_input(telemetry_fixtures: Path) -> None:
    reading = parse_meminfo(_fixture(telemetry_fixtures, "proc/meminfo-minimal.txt"))

    assert reading.total_bytes == 2048 * 1024
    assert reading.available_bytes == 512 * 1024
    assert reading.used_bytes == 1536 * 1024


def test_parse_meminfo_documents_missing_memavailable_fallback(telemetry_fixtures: Path) -> None:
    reading = parse_meminfo(_fixture(telemetry_fixtures, "proc/meminfo-fallback.txt"))

    assert reading.available_bytes == 340 * 1024
    assert reading.used_bytes == 660 * 1024


@pytest.mark.parametrize(
    ("text", "total", "available", "used"),
    [
        ("garbage\n", UNSUPPORTED, UNSUPPORTED, UNSUPPORTED),
        ("MemTotal: nope kB\nMemAvailable: 2 kB\n", UNSUPPORTED, 2048, UNSUPPORTED),
        ("MemTotal: 1 MB\nMemAvailable: 1 kB\n", UNSUPPORTED, 1024, UNSUPPORTED),
        ("MemTotal: 1 kB\nMemAvailable: 2 kB\n", 1024, 2048, UNSUPPORTED),
        ("MemTotal: -1 kB\n", UNSUPPORTED, UNSUPPORTED, UNSUPPORTED),
    ],
)
def test_parse_meminfo_malformed_fields_degrade_independently(
    text: str, total: object, available: object, used: object
) -> None:
    reading = parse_meminfo(text)

    assert reading.total_bytes == total
    assert reading.available_bytes == available
    assert reading.used_bytes == used


def test_parse_meminfo_empty_value_and_fallback_without_total() -> None:
    reading = parse_meminfo("MemTotal:\nMemFree: 10 kB\nCached: 5 kB\n")

    assert reading.total_bytes is UNSUPPORTED
    assert reading.available_bytes == 15 * 1024
    assert reading.used_bytes is UNSUPPORTED


def test_memory_read_failure_returns_all_unsupported() -> None:
    def denied(_path: Path) -> str:
        raise PermissionError("fixture denied")

    reading = LinuxHostReader(read_text=denied).memory()

    assert reading.total_bytes is UNSUPPORTED
    assert reading.available_bytes is UNSUPPORTED
    assert reading.used_bytes is UNSUPPORTED


def test_parse_loadavg_reads_one_minute_value(telemetry_fixtures: Path) -> None:
    assert parse_loadavg(_fixture(telemetry_fixtures, "proc/loadavg.txt")) == 1.25


@pytest.mark.parametrize("text", ["", "not-a-number 0 0", "nan 0 0", "inf 0 0", "-1 0 0"])
def test_parse_loadavg_malformed_is_unsupported(text: str) -> None:
    assert parse_loadavg(text) is UNSUPPORTED


def test_parse_diskstats_sums_whole_devices_without_partitions_or_virtuals(
    telemetry_fixtures: Path,
) -> None:
    parsed = parse_diskstats(_fixture(telemetry_fixtures, "proc/diskstats-first.txt"))

    assert parsed == (6000 * 512, 2500 * 512)


def test_parse_diskstats_honors_exact_device_filter(telemetry_fixtures: Path) -> None:
    parsed = parse_diskstats(
        _fixture(telemetry_fixtures, "proc/diskstats-first.txt"), devices={"sda"}
    )

    assert parsed == (2000 * 512, 1000 * 512)


@pytest.mark.parametrize(
    ("text", "devices"),
    [
        ("garbage\n", None),
        ("8 0 sda 1 2\n", None),
        ("8 0 sda 1 0 bad 0 1 0 2 0\n", None),
        ("8 0 sda 1 0 -1 0 1 0 2 0\n", None),
        ("8 0 sda 1 0 2 0 1 0 2 0\n", {"sda", "sdb"}),
        ("8 0 loop0 1 0 2 0 1 0 2 0\n", None),
    ],
)
def test_parse_diskstats_malformed_or_incomplete_is_unsupported(
    text: str, devices: set[str] | None
) -> None:
    assert parse_diskstats(text, devices=devices) is UNSUPPORTED


def test_disk_throughput_first_call_then_rate_uses_elapsed_clock(telemetry_fixtures: Path) -> None:
    samples = iter(
        [
            _fixture(telemetry_fixtures, "proc/diskstats-first.txt"),
            _fixture(telemetry_fixtures, "proc/diskstats-second.txt"),
        ]
    )
    times = iter([10.0, 12.0])
    reader = LinuxHostReader(read_text=lambda _path: next(samples), clock=lambda: next(times))

    first = reader.disk_throughput()
    second = reader.disk_throughput()

    assert first.read_bytes_per_sec is UNSUPPORTED
    assert first.write_bytes_per_sec is UNSUPPORTED
    assert second.read_bytes_per_sec == 2000 * 512 / 2
    assert second.write_bytes_per_sec == 2000 * 512 / 2


def test_disk_counter_rollback_degrades_only_affected_direction_and_resets() -> None:
    samples = iter(
        [
            "8 0 sda 1 0 2000 0 1 0 1000 0\n",
            "8 0 sda 1 0 100 0 1 0 2000 0\n",
            "8 0 sda 1 0 200 0 1 0 3000 0\n",
        ]
    )
    times = iter([0.0, 1.0, 2.0])
    reader = LinuxHostReader(read_text=lambda _path: next(samples), clock=lambda: next(times))

    reader.disk_throughput()
    wrapped = reader.disk_throughput()
    recovered = reader.disk_throughput()

    assert wrapped.read_bytes_per_sec is UNSUPPORTED
    assert wrapped.write_bytes_per_sec == 1000 * 512
    assert recovered.read_bytes_per_sec == 100 * 512
    assert recovered.write_bytes_per_sec == 1000 * 512


@pytest.mark.parametrize("times", [[1.0, 1.0], [2.0, 1.0], [1.0, float("nan")]])
def test_disk_throughput_invalid_elapsed_time_is_unsupported(times: list[float]) -> None:
    samples = iter(["8 0 sda 1 0 10 0 1 0 10 0\n", "8 0 sda 1 0 20 0 1 0 20 0\n"])
    clock_values = iter(times)
    reader = LinuxHostReader(
        read_text=lambda _path: next(samples), clock=lambda: next(clock_values)
    )

    reader.disk_throughput()
    reading = reader.disk_throughput()

    assert reading.read_bytes_per_sec is UNSUPPORTED
    assert reading.write_bytes_per_sec is UNSUPPORTED


def test_parse_cpuinfo_derives_multisocket_physical_and_logical_counts(
    telemetry_fixtures: Path,
) -> None:
    facts = parse_cpuinfo(_fixture(telemetry_fixtures, "proc/cpuinfo-x86.txt"))

    assert facts.cpu_model == "Dual Socket Test CPU"
    assert facts.physical_cores == 3
    assert facts.logical_cores == 4


def test_parse_cpuinfo_arm_uses_hardware_model_without_guessing_physical_cores(
    telemetry_fixtures: Path,
) -> None:
    facts = parse_cpuinfo(_fixture(telemetry_fixtures, "proc/cpuinfo-arm.txt"))

    assert facts.cpu_model == "BCM2711"
    assert facts.physical_cores is UNSUPPORTED
    assert facts.logical_cores == 2


@pytest.mark.parametrize("text", ["", "garbage\n", "processor:\nmodel name:\n"])
def test_parse_cpuinfo_malformed_fields_are_unsupported(text: str) -> None:
    facts = parse_cpuinfo(text)

    assert facts.cpu_model is UNSUPPORTED
    assert facts.physical_cores is UNSUPPORTED
    assert facts.logical_cores is UNSUPPORTED


def test_read_thermal_zones_prefers_cpu_package_sensor(telemetry_fixtures: Path) -> None:
    temperature = read_thermal_zones(telemetry_fixtures / "sys/class/thermal")

    assert temperature == 55.5


def test_read_thermal_zones_skips_malformed_and_permission_denied_files(tmp_path: Path) -> None:
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()

    def denied_or_malformed(path: Path) -> str:
        if path.name == "type":
            raise PermissionError("fixture denied")
        return "not-a-temperature"

    assert read_thermal_zones(tmp_path, read_text=denied_or_malformed) is UNSUPPORTED


def test_read_thermal_zones_accepts_unknown_type_as_last_resort(tmp_path: Path) -> None:
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("mystery_sensor\n", encoding="utf-8")
    (zone / "temp").write_text("42000\n", encoding="utf-8")

    assert read_thermal_zones(tmp_path) == 42.0


@pytest.mark.parametrize("raw", ["nan", "-274000", "1001000"])
def test_thermal_zones_reject_nonfinite_or_implausible_values(tmp_path: Path, raw: str) -> None:
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("x86_pkg_temp\n", encoding="utf-8")
    (zone / "temp").write_text(raw, encoding="utf-8")

    assert read_thermal_zones(tmp_path) is UNSUPPORTED


def test_thermal_zone_missing_temperature_is_skipped(tmp_path: Path) -> None:
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("x86_pkg_temp\n", encoding="utf-8")

    assert read_thermal_zones(tmp_path) is UNSUPPORTED


def test_read_hwmon_uses_cpu_sensor_and_ignores_gpu_sensor(telemetry_fixtures: Path) -> None:
    temperature = read_hwmon(telemetry_fixtures / "sys/class/hwmon")

    assert temperature == 47.5


def test_read_hwmon_skips_malformed_temperature(tmp_path: Path) -> None:
    hwmon = tmp_path / "hwmon0"
    hwmon.mkdir()
    (hwmon / "name").write_text("coretemp\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text("broken\n", encoding="utf-8")

    assert read_hwmon(tmp_path) is UNSUPPORTED


def test_linux_reader_falls_back_to_hwmon_when_thermal_tree_absent(
    telemetry_fixtures: Path, tmp_path: Path
) -> None:
    shutil.copytree(telemetry_fixtures / "sys/class/hwmon", tmp_path / "class/hwmon")
    reader = LinuxHostReader(sys_root=tmp_path)

    assert reader.cpu_temperature() == 47.5


def test_cpu_temperature_returns_unsupported_when_neither_tree_exists(tmp_path: Path) -> None:
    assert LinuxHostReader(sys_root=tmp_path).cpu_temperature() is UNSUPPORTED


def test_linux_reader_returns_thermal_zone_without_consulting_hwmon(
    telemetry_fixtures: Path,
) -> None:
    reader = LinuxHostReader(sys_root=telemetry_fixtures / "sys")

    assert reader.cpu_temperature() == 55.5


def test_linux_reader_caches_sensor_path_and_rediscovers_after_invalid_read(
    tmp_path: Path,
) -> None:
    zone = tmp_path / "class" / "thermal" / "thermal_zone0"
    zone.mkdir(parents=True)
    temperatures = iter(["50000", "51000", "broken", "52000"])
    reads: list[str] = []

    def scripted_read(path: Path) -> str:
        reads.append(path.name)
        if path.name == "type":
            return "x86_pkg_temp"
        return next(temperatures)

    reader = LinuxHostReader(sys_root=tmp_path, read_text=scripted_read)

    assert reader.cpu_temperature() == 50.0
    assert reader.cpu_temperature() == 51.0
    assert reader.cpu_temperature() == 52.0
    assert reads.count("type") == 2
    assert reads.count("temp") == 4


def test_read_block_devices_normalizes_attributes_and_filters_virtuals(
    telemetry_fixtures: Path,
) -> None:
    devices = read_block_devices(telemetry_fixtures / "sys/block")

    assert [device.name for device in devices] == ["nvme0n1", "sda"]
    assert devices[0].size_bytes == 1_000_215_216 * 512
    assert devices[0].model == "Fixture NVMe"
    assert devices[0].rotational is False
    assert devices[1].rotational is True


def test_read_block_devices_unreadable_attributes_degrade_independently(tmp_path: Path) -> None:
    (tmp_path / "sda").mkdir()

    devices = read_block_devices(tmp_path)

    assert len(devices) == 1
    assert devices[0].size_bytes is UNSUPPORTED
    assert devices[0].model is None
    assert devices[0].rotational is None


def test_read_block_devices_malformed_size_and_rotational_are_unavailable(tmp_path: Path) -> None:
    device = tmp_path / "sda"
    (device / "queue").mkdir(parents=True)
    (device / "size").write_text("not-a-number\n", encoding="utf-8")
    (device / "queue/rotational").write_text("maybe\n", encoding="utf-8")

    result = read_block_devices(tmp_path)[0]

    assert result.size_bytes is UNSUPPORTED
    assert result.rotational is None


def test_static_facts_uses_injected_proc_and_sys_roots(
    telemetry_fixtures: Path, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "cpuinfo").write_text(
        _fixture(telemetry_fixtures, "proc/cpuinfo-x86.txt"), encoding="utf-8"
    )
    (proc_root / "meminfo").write_text(
        _fixture(telemetry_fixtures, "proc/meminfo-minimal.txt"), encoding="utf-8"
    )
    reader: HostReader = LinuxHostReader(proc_root=proc_root, sys_root=telemetry_fixtures / "sys")

    facts = reader.static_facts()

    assert facts.cpu_model == "Dual Socket Test CPU"
    assert facts.physical_cores == 3
    assert facts.logical_cores == 4
    assert facts.ram_bytes == 2048 * 1024
    assert [device.name for device in facts.storage] == ["nvme0n1", "sda"]


def test_reported_system_text_degrades_empty_and_failed_values() -> None:
    def fail_system() -> str:
        raise OSError("uname failed")

    assert _reported_text(lambda: "") is UNSUPPORTED
    assert _reported_text(fail_system) is UNSUPPORTED


def test_safe_returns_result_without_logging(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="sweatmeter.safe"):
        result = _safe(lambda: 42)

    assert result == 42
    assert not caplog.records


def test_safe_returns_default_and_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    def fail() -> int:
        raise PermissionError("fixture denied")

    with caplog.at_level(logging.DEBUG, logger="sweatmeter.safe"):
        result = _safe(fail, -1)

    assert result == -1
    record = caplog.records[0]
    assert record.getMessage() == "telemetry.source_failed"
    assert getattr(record, "operation", None) == fail.__qualname__
    assert "fixture denied" in caplog.text


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit()])
def test_safe_does_not_catch_process_control_exceptions(exception: BaseException) -> None:
    def interrupt() -> int:
        raise exception

    with pytest.raises(type(exception)):
        _safe(interrupt)


def test_linux_reader_unreadable_load_and_clock_are_unsupported() -> None:
    def denied(_path: Path) -> str:
        raise PermissionError("fixture denied")

    def broken_clock() -> float:
        raise RuntimeError("clock failed")

    reader = LinuxHostReader(read_text=denied, clock=broken_clock)

    assert reader.load_average_1m() is UNSUPPORTED
    assert reader.disk_throughput().read_bytes_per_sec is UNSUPPORTED


def test_disk_counter_write_rollback_degrades_only_write_direction() -> None:
    samples = iter(
        [
            "8 0 sda 1 0 100 0 1 0 2000 0\n",
            "8 0 sda 1 0 200 0 1 0 100 0\n",
        ]
    )
    times = iter([0.0, 1.0])
    reader = LinuxHostReader(read_text=lambda _path: next(samples), clock=lambda: next(times))

    reader.disk_throughput()
    wrapped = reader.disk_throughput()

    assert wrapped.read_bytes_per_sec == 100 * 512
    assert wrapped.write_bytes_per_sec is UNSUPPORTED


@pytest.mark.parametrize(
    "parser",
    [
        parse_proc_stat,
        parse_meminfo,
        parse_loadavg,
        parse_diskstats,
        parse_cpuinfo,
    ],
)
def test_all_text_parsers_tolerate_hostile_fixture_text(parser: Callable[[str], object]) -> None:
    parser("\x00\udcff: not-a-number kB\n" * 10)


def test_public_type_aliases_and_protocol_annotations_resolve_at_runtime() -> None:
    assert ReportedText.__value__ == str | Unsupported
    assert get_type_hints(MemoryReading)["total_bytes"] is Measurement
    assert get_type_hints(HostReader.memory)["return"] is MemoryReading
