"""One conformance suite run against every ``GpuReader`` implementation.

ADR-0021 §7 requires that a second GPU backend passes the same suite as the first, and Testing
Standards §7 requires a port's conformance suite to run against every implementation — real, fake
and recorded. These tests therefore assert only what the *protocol* promises, never what one
backend happens to do, and the final test pins the two NVIDIA backends to identical output for the
same device so they are genuinely interchangeable rather than merely similar.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from baseaicore import GpuProfile, GpuVendor, is_supported

from conftest import FakeDevice, FakeNvml
from sweatmeter import GpuReader, GpuSample, NullGpuReader, NvidiaSmiReader, NvmlGpuReader
from sweatmeter.testing import ScriptedGpuReader

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "telemetry" / "nvidia"
_DRIVER = "580.173.02"

# Every reason a `GpuSample` may name. The two NVIDIA backends must agree on this vocabulary or
# `TelemetryWindow`'s throttle heuristic would mean different things depending on the backend.
_THROTTLE_VOCABULARY = frozenset(
    {
        "gpu_idle",
        "applications_clocks_setting",
        "sw_power_cap",
        "hw_slowdown",
        "sync_boost",
        "sw_thermal_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
)
_PERCENT_FIELDS = ("utilization_percent", "memory_utilization_percent", "fan_percent")
_TEMPERATURE_FIELDS = ("temperature_c", "memory_temperature_c")
_NON_NEGATIVE_FIELDS = (
    "vram_used_bytes",
    "vram_total_bytes",
    "power_watts",
    "power_limit_watts",
    "core_clock_mhz",
    "memory_clock_mhz",
)
_MEASUREMENT_FIELDS = _PERCENT_FIELDS + _TEMPERATURE_FIELDS + _NON_NEGATIVE_FIELDS


def _fixture(name: str) -> str:
    return (_FIXTURES / f"{_DRIVER}-{name}.csv").read_text(encoding="utf-8")


class _ReplayRunner:
    """Answer each ``nvidia-smi`` query from the committed fixture for that query."""

    def __init__(self, *, sample: str, throttle: str, static: str, compute: str) -> None:
        self._by_field = {
            "clocks_throttle_reasons": throttle,
            "compute_cap": compute,
            "driver_version": static,
            "utilization.gpu": sample,
        }

    def __call__(  # noqa: PLR0913 — mirrors the production runner protocol exactly
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        check: bool,
        encoding: str,
        errors: str,
        shell: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, check, encoding, errors, shell, text, timeout
        query = args[1] if len(args) > 1 else ""
        # `available()` probes with the index column alone; the sample fixture leads with it.
        output = self._by_field["utilization.gpu"] if query.endswith("=index") else ""
        for marker, fixture in self._by_field.items():
            if marker in query:
                output = fixture
                break
        return subprocess.CompletedProcess(list(args), 0, output, "")


def _recorded_nvidia_smi(*, devices: str = "single") -> NvidiaSmiReader:
    """Build the command backend replaying real captured driver output."""
    return NvidiaSmiReader(
        runner=_ReplayRunner(
            sample=_fixture(f"{devices}-sample" if devices == "single" else "two-gpu-sample"),
            throttle=_fixture(f"{devices}-throttle" if devices == "single" else "two-gpu-throttle"),
            static=_fixture("single-static"),
            compute=_fixture("single-compute"),
        ),
        resolver=lambda _name: "/opt/nvidia/bin/nvidia-smi",
    )


def _faked_nvml(*, devices: int = 1) -> NvmlGpuReader:
    """Build the NVML backend over a fake binding with the given device count."""
    return NvmlGpuReader(
        binding=FakeNvml(devices=[FakeDevice(uuid=f"GPU-{index}") for index in range(devices)])
    )


def _scripted() -> ScriptedGpuReader:
    return ScriptedGpuReader([[GpuSample(index=0, vram_used_bytes=1_024)]] * 4)


# Every implementation of the port, in the three flavours the testing standards name: recorded
# (real captured driver output), fake (in-memory doubles) and the deliberately empty null reader.
_IMPLEMENTATIONS: dict[str, Callable[[], GpuReader]] = {
    "nvidia_smi_recorded": _recorded_nvidia_smi,
    "nvidia_smi_recorded_two_gpu": lambda: _recorded_nvidia_smi(devices="two"),
    "nvml_fake": _faked_nvml,
    "nvml_fake_two_gpu": lambda: _faked_nvml(devices=2),
    "null": NullGpuReader,
    "scripted": _scripted,
}

_reader_case = pytest.mark.parametrize(
    "build_reader", list(_IMPLEMENTATIONS.values()), ids=list(_IMPLEMENTATIONS)
)


@pytest.mark.contract
@_reader_case
def test_available_answers_with_a_bool(build_reader: Callable[[], GpuReader]) -> None:
    assert isinstance(build_reader().available(), bool)


@pytest.mark.contract
@_reader_case
def test_sample_returns_gpu_samples(build_reader: Callable[[], GpuReader]) -> None:
    samples = build_reader().sample()

    assert isinstance(samples, Sequence)
    assert all(isinstance(sample, GpuSample) for sample in samples)


@pytest.mark.contract
@_reader_case
def test_static_info_returns_nvidia_profiles(build_reader: Callable[[], GpuReader]) -> None:
    profiles = build_reader().static_info()

    assert isinstance(profiles, Sequence)
    for profile in profiles:
        assert isinstance(profile, GpuProfile)
        assert isinstance(profile.vendor, GpuVendor)
        assert profile.name is None or profile.name.strip()
        assert profile.uuid is None or profile.uuid.strip()


@pytest.mark.contract
@_reader_case
def test_device_indices_are_non_negative_unique_and_ordered(
    build_reader: Callable[[], GpuReader],
) -> None:
    reader = build_reader()

    for indices in ([s.index for s in reader.sample()], [p.index for p in reader.static_info()]):
        assert all(index >= 0 for index in indices)
        assert len(set(indices)) == len(indices)
        assert indices == sorted(indices)


@pytest.mark.contract
@_reader_case
def test_every_measurement_is_a_number_or_unsupported_never_a_placeholder(
    build_reader: Callable[[], GpuReader],
) -> None:
    for sample in build_reader().sample():
        for name in _MEASUREMENT_FIELDS:
            value = getattr(sample, name)
            if is_supported(value):
                assert isinstance(value, (int, float)) and not isinstance(value, bool), name
            else:
                # Unavailable is the sentinel, never None, "", "N/A" or a fabricated zero.
                assert value is not None, name


@pytest.mark.contract
@_reader_case
def test_units_stay_inside_their_documented_ranges(
    build_reader: Callable[[], GpuReader],
) -> None:
    for sample in build_reader().sample():
        for name in _PERCENT_FIELDS:
            value = getattr(sample, name)
            if is_supported(value):
                assert 0 <= value <= 100, f"{name}={value!r}"
        for name in _TEMPERATURE_FIELDS:
            value = getattr(sample, name)
            if is_supported(value):
                assert value >= -273.15, f"{name}={value!r}"
        for name in _NON_NEGATIVE_FIELDS:
            value = getattr(sample, name)
            if is_supported(value):
                assert value >= 0, f"{name}={value!r}"


@pytest.mark.contract
@_reader_case
def test_throttle_reporting_is_honest_about_what_it_knows(
    build_reader: Callable[[], GpuReader],
) -> None:
    for sample in build_reader().sample():
        assert isinstance(sample.throttle_reasons_available, bool)
        assert isinstance(sample.throttle_reasons, tuple)
        assert set(sample.throttle_reasons) <= _THROTTLE_VOCABULARY
        if not sample.throttle_reasons_available:
            # "Cannot tell" must never masquerade as "nothing is throttling".
            assert sample.throttle_reasons == ()


@pytest.mark.contract
@_reader_case
def test_a_reader_reporting_devices_also_reports_itself_available(
    build_reader: Callable[[], GpuReader],
) -> None:
    reader = build_reader()

    if reader.sample():
        assert reader.available() is True


@pytest.mark.contract
@_reader_case
def test_repeated_collection_is_stable_and_non_raising(
    build_reader: Callable[[], GpuReader],
) -> None:
    reader = build_reader()

    first = [sample.index for sample in reader.sample()]
    second = [sample.index for sample in reader.sample()]

    assert first == second


@pytest.mark.contract
def test_both_nvidia_backends_describe_the_same_device_identically() -> None:
    """The two NVIDIA backends must be swappable, not merely similar.

    The fake NVML device is configured with the values captured in the committed driver fixture,
    so any disagreement here is a real unit or semantics divergence between the backends rather
    than a difference in the hardware they were pointed at.
    """
    mib = 1024 * 1024
    nvml = NvmlGpuReader(
        binding=FakeNvml(
            devices=[
                FakeDevice(
                    uuid="GPU-fixture-0000",
                    util_gpu=2,
                    util_memory=10,
                    memory_used=655 * mib,
                    memory_total=16_311 * mib,
                    temperature=29,
                    power_milliwatts=9_990,
                    power_limit_milliwatts=180_000,
                    fan=0,
                    clock_sm=570,
                    clock_mem=405,
                )
            ]
        )
    )

    from_command = _recorded_nvidia_smi().sample()[0]
    from_nvml = nvml.sample()[0]

    assert from_command.index == from_nvml.index
    assert from_command.uuid == from_nvml.uuid
    for name in _MEASUREMENT_FIELDS:
        command_value, nvml_value = getattr(from_command, name), getattr(from_nvml, name)
        if is_supported(command_value) or is_supported(nvml_value):
            assert command_value == pytest.approx(nvml_value), name
        else:
            assert not is_supported(command_value) and not is_supported(nvml_value), name
    assert from_command.throttle_reasons == from_nvml.throttle_reasons
    assert from_command.throttle_reasons_available == from_nvml.throttle_reasons_available
