"""Opt-in shape and plausibility checks against the machine running the suite."""

from __future__ import annotations

import time

import pytest
from baseaicore import is_supported

from sweatmeter import (
    LinuxHostReader,
    NullGpuReader,
    NvidiaSmiReader,
    NvmlGpuReader,
    TelemetryCollector,
    nvml_binding_available,
)


@pytest.mark.live
def test_real_linux_host_snapshot_and_profile_are_plausible() -> None:
    collector = TelemetryCollector(host=LinuxHostReader(), gpu=NullGpuReader())
    collector.snapshot()  # Prime cumulative CPU and disk counters.
    time.sleep(0.01)

    snapshot = collector.snapshot()
    profile = collector.machine_profile()

    assert snapshot.timestamp.tzinfo is not None
    assert is_supported(snapshot.cpu_percent)
    assert 0 <= snapshot.cpu_percent <= 100
    assert is_supported(snapshot.ram_total_bytes)
    assert snapshot.ram_total_bytes > 0
    assert is_supported(snapshot.process_rss_bytes)
    assert snapshot.process_rss_bytes > 0
    assert len(profile.machine_fingerprint) == 64
    assert profile.os_name == "Linux"


@pytest.mark.live
def test_real_nvidia_gpu_is_plausible_when_available() -> None:
    reader = NvidiaSmiReader()
    if not reader.available():
        pytest.skip("nvidia-smi did not report a GPU on this live-test machine")

    samples = reader.sample()
    profiles = reader.static_info()

    assert samples
    assert profiles
    assert [sample.index for sample in samples] == sorted(sample.index for sample in samples)
    assert all(
        not is_supported(sample.utilization_percent) or 0 <= sample.utilization_percent <= 100
        for sample in samples
    )
    assert all(profile.index >= 0 for profile in profiles)


@pytest.mark.live
def test_real_nvml_backend_agrees_with_the_command_backend() -> None:
    if not nvml_binding_available():
        pytest.skip("the optional pynvml extra is not installed on this live-test machine")
    command = NvidiaSmiReader()
    if not command.available():
        pytest.skip("nvidia-smi did not report a GPU on this live-test machine")

    with NvmlGpuReader() as nvml:
        assert nvml.available() is True
        nvml_profiles = nvml.static_info()
        nvml_samples = nvml.sample()
    command_profiles = command.static_info()

    # Identity must match exactly: the two backends read the same devices through different APIs,
    # so a disagreement here would mean a consumer's evidence depends on which one was installed.
    assert [profile.uuid for profile in nvml_profiles] == [
        profile.uuid for profile in command_profiles
    ]
    assert [profile.name for profile in nvml_profiles] == [
        profile.name for profile in command_profiles
    ]
    assert [profile.vram_total_bytes for profile in nvml_profiles] == [
        profile.vram_total_bytes for profile in command_profiles
    ]
    for sample in nvml_samples:
        assert is_supported(sample.vram_used_bytes)
        assert is_supported(sample.temperature_c)
        assert 0 <= sample.utilization_percent <= 100
