"""Factory, tier-3 stub, null-reader, and process-RSS tests for Phase 3."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from baseaicore import (
    UNSUPPORTED,
    DependencyUnavailableError,
    UnsupportedPlatformError,
    ValidationError,
)

import sweatmeter
from sweatmeter import (
    DarwinHostReader,
    GpuBackend,
    HostReader,
    LinuxHostReader,
    NullGpuReader,
    NullHostReader,
    NvidiaSmiReader,
    NvmlGpuReader,
    WindowsHostReader,
    create_gpu_reader,
    create_host_reader,
    nvml_binding_available,
)
from sweatmeter import platform as platform_module


@pytest.mark.parametrize(
    ("platform_name", "expected_type"),
    [
        ("linux", LinuxHostReader),
        ("win32", WindowsHostReader),
        ("darwin", DarwinHostReader),
    ],
)
def test_host_factory_selects_every_documented_platform_branch(
    platform_name: str, expected_type: type[object]
) -> None:
    assert isinstance(create_host_reader(platform_name=platform_name), expected_type)


@pytest.mark.linux_only
def test_host_factory_uses_running_platform_by_default() -> None:
    assert isinstance(create_host_reader(), LinuxHostReader)


def test_host_factory_rejects_unknown_platform_with_degradation_guidance() -> None:
    with pytest.raises(UnsupportedPlatformError, match="NullHostReader") as raised:
        create_host_reader(platform_name="plan9")

    assert raised.value.details == {"platform": "plan9", "feature": "host telemetry"}


def test_gpu_factory_honours_an_explicit_backend_without_probing_hardware() -> None:
    assert isinstance(create_gpu_reader(prefer=GpuBackend.NVIDIA_SMI), NvidiaSmiReader)


def test_gpu_factory_default_follows_optional_binding_availability() -> None:
    expected = NvmlGpuReader if nvml_binding_available() else NvidiaSmiReader

    assert isinstance(create_gpu_reader(), expected)


def test_gpu_factory_prefers_nvml_when_the_extra_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulating presence/absence of an installed distribution is a boundary condition; the
    # alternative would be uninstalling a package inside a test run.
    monkeypatch.setattr(platform_module, "nvml_binding_available", lambda: True)

    assert isinstance(create_gpu_reader(), NvmlGpuReader)


def test_gpu_factory_falls_back_to_the_command_backend_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "nvml_binding_available", lambda: False)

    assert isinstance(create_gpu_reader(), NvidiaSmiReader)


def test_explicitly_requested_nvml_backend_is_honoured_when_the_extra_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "nvml_binding_available", lambda: True)

    assert isinstance(create_gpu_reader(prefer=GpuBackend.PYNVML), NvmlGpuReader)


def test_explicitly_requested_nvml_backend_is_refused_when_the_extra_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "nvml_binding_available", lambda: False)

    with pytest.raises(DependencyUnavailableError, match="sweatmeter\\[pynvml\\]"):
        create_gpu_reader(prefer=GpuBackend.PYNVML)


def test_gpu_factory_rejects_runtime_value_outside_backend_enum() -> None:
    with pytest.raises(ValidationError, match="Unsupported GPU backend"):
        create_gpu_reader(prefer=cast(GpuBackend, "future-backend"))


@pytest.mark.parametrize("reader_type", [WindowsHostReader, DarwinHostReader])
@pytest.mark.parametrize(
    "operation",
    [
        "cpu_percent",
        "load_average_1m",
        "memory",
        "cpu_temperature",
        "disk_throughput",
        "process_rss_bytes",
        "static_facts",
    ],
)
def test_tier_three_stub_operations_raise_instead_of_returning_zero(
    reader_type: type[WindowsHostReader] | type[DarwinHostReader], operation: str
) -> None:
    reader = reader_type()
    method = cast(Callable[[], object], getattr(reader, operation))

    with pytest.raises(UnsupportedPlatformError, match="tier-3"):
        method()


def test_null_host_reader_is_protocol_compatible_and_never_returns_zero() -> None:
    reader = NullHostReader()
    protocol_reader: HostReader = reader

    assert isinstance(reader, HostReader)
    assert protocol_reader.cpu_percent() is UNSUPPORTED
    assert protocol_reader.load_average_1m() is UNSUPPORTED
    assert protocol_reader.memory().total_bytes is UNSUPPORTED
    assert protocol_reader.cpu_temperature() is UNSUPPORTED
    assert protocol_reader.disk_throughput().read_bytes_per_sec is UNSUPPORTED
    assert protocol_reader.process_rss_bytes() is UNSUPPORTED
    assert protocol_reader.static_facts().ram_bytes is UNSUPPORTED
    assert set(reader.unavailable_reasons().values()) == {"platform_unsupported"}


def test_null_gpu_reader_is_empty_unavailable_and_reasoned() -> None:
    reader = NullGpuReader()

    assert reader.available() is False
    assert reader.sample() == ()
    assert reader.static_info() == ()
    assert reader.unavailable_reasons() == {"gpu": "platform_unsupported"}


def test_null_readers_reject_blank_reason() -> None:
    with pytest.raises(ValidationError, match="reason must be non-empty"):
        NullHostReader(reason="  ")
    with pytest.raises(ValidationError, match="reason must be non-empty"):
        NullGpuReader(reason="  ")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Name:\tpython\nVmRSS:\t1234 kB\nThreads:\t1\n", 1234 * 1024),
        ("VmRSS: 0 kB\n", 0),
        ("VmRSS: malformed kB\n", UNSUPPORTED),
        ("VmRSS: 1234 MB\n", UNSUPPORTED),
        ("Name:\tpython\n", UNSUPPORTED),
    ],
)
def test_linux_process_rss_reader_parses_current_resident_bytes(
    status: str, expected: object
) -> None:
    reader = LinuxHostReader(read_text=lambda _path: status)

    assert reader.process_rss_bytes() == expected


def test_linux_process_rss_reader_degrades_permission_error() -> None:
    def denied(_path: Path) -> str:
        raise PermissionError("fixture denied")

    assert LinuxHostReader(read_text=denied).process_rss_bytes() is UNSUPPORTED


def test_phase_three_public_api_is_exported_from_package_root() -> None:
    expected = {
        "GpuBackend",
        "NullGpuReader",
        "NullHostReader",
        "TelemetryCollector",
        "TelemetrySampler",
        "TelemetrySnapshot",
        "create_gpu_reader",
        "create_host_reader",
    }

    assert expected <= set(sweatmeter.__all__)
