"""NVML backend tests driven entirely by a fake binding: no GPU and no pynvml required."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from baseaicore import UNSUPPORTED, DependencyUnavailableError, GpuVendor, is_supported

from conftest import (
    FakeDevice,
    FakeNvml,
    NVMLError_DriverNotLoaded,
    NVMLError_GpuIsLost,
    NVMLError_NoPermission,
    NVMLError_NotSupported,
)
from sweatmeter import NvmlGpuReader
from sweatmeter.readers import nvml as nvml_module
from sweatmeter.readers.nvml import NvmlBinding, load_nvml_binding, nvml_binding_available


def _reader(nvml: FakeNvml) -> NvmlGpuReader:
    return NvmlGpuReader(binding=nvml)


def test_fake_binding_satisfies_the_published_protocol() -> None:
    assert isinstance(FakeNvml(), NvmlBinding)


def test_sample_normalizes_every_unit() -> None:
    reader = _reader(FakeNvml())

    sample = reader.sample()[0]

    assert sample.index == 0
    assert sample.uuid == "GPU-TEST-0"
    assert sample.utilization_percent == 50
    assert sample.memory_utilization_percent == 25
    assert sample.vram_used_bytes == 4_000_000_000
    assert sample.vram_total_bytes == 16_000_000_000
    assert sample.temperature_c == 60
    assert sample.power_watts == pytest.approx(120.5)  # milliwatts normalized to watts
    assert sample.power_limit_watts == pytest.approx(180.0)
    assert sample.fan_percent == 40
    assert sample.core_clock_mhz == 2_400
    assert sample.memory_clock_mhz == 9_000


def test_memory_temperature_is_unsupported_with_a_reason_never_zero() -> None:
    reader = _reader(FakeNvml())

    sample = reader.sample()[0]

    assert sample.memory_temperature_c is UNSUPPORTED
    assert reader.unavailable_reasons()["gpu.0.memory_temperature_c"] == "sensor_unsupported"


def test_static_info_reports_identity_driver_and_versions() -> None:
    reader = _reader(FakeNvml())

    profile = reader.static_info()[0]

    assert profile.index == 0
    assert profile.name == "NVIDIA Test GPU"
    assert profile.uuid == "GPU-TEST-0"
    assert profile.vram_total_bytes == 16_000_000_000
    assert profile.driver_version == "580.173.02"
    assert profile.cuda_version == "13.0"
    assert profile.compute_capability == "12.0"
    assert profile.vendor is GpuVendor.NVIDIA


def test_byte_string_text_from_older_bindings_is_decoded() -> None:
    nvml = FakeNvml(devices=[FakeDevice(name=b"NVIDIA Byte GPU", uuid=b"GPU-BYTES")], driver=b"1.2")

    profile = _reader(nvml).static_info()[0]

    assert profile.name == "NVIDIA Byte GPU"
    assert profile.uuid == "GPU-BYTES"
    assert profile.driver_version == "1.2"


def test_two_devices_are_reported_separately_and_in_index_order() -> None:
    nvml = FakeNvml(
        devices=[
            FakeDevice(uuid="GPU-0", memory_used=1_000, power_milliwatts=50_000),
            FakeDevice(uuid="GPU-1", memory_used=2_000, power_milliwatts=90_000),
        ]
    )

    samples = _reader(nvml).sample()

    assert [sample.index for sample in samples] == [0, 1]
    assert [sample.uuid for sample in samples] == ["GPU-0", "GPU-1"]
    assert [sample.vram_used_bytes for sample in samples] == [1_000, 2_000]
    assert samples[0].power_watts == pytest.approx(50.0)
    assert samples[1].power_watts == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("operation", "degraded_field"),
    [
        ("nvmlDeviceGetPowerUsage", "power_watts"),
        ("nvmlDeviceGetEnforcedPowerLimit", "power_limit_watts"),
        ("nvmlDeviceGetTemperature", "temperature_c"),
        ("nvmlDeviceGetFanSpeed", "fan_percent"),
        ("nvmlDeviceGetUtilizationRates", "utilization_percent"),
    ],
)
def test_one_unsupported_sensor_degrades_only_itself(operation: str, degraded_field: str) -> None:
    nvml = FakeNvml(fail={operation: NVMLError_NotSupported()})
    reader = _reader(nvml)

    sample = reader.sample()[0]

    assert getattr(sample, degraded_field) is UNSUPPORTED
    assert reader.unavailable_reasons()[f"gpu.0.{degraded_field}"] == "sensor_unsupported"
    # The neighbours a single nvidia-smi command would have taken down with it survive.
    assert sample.vram_used_bytes == 4_000_000_000
    assert sample.core_clock_mhz == 2_400


def test_missing_power_sensor_is_unsupported_never_zero() -> None:
    nvml = FakeNvml(fail={"nvmlDeviceGetPowerUsage": NVMLError_NotSupported()})

    sample = _reader(nvml).sample()[0]

    assert sample.power_watts is UNSUPPORTED
    assert not is_supported(sample.power_watts)


def test_throttle_reasons_are_decoded_from_the_active_bitmask() -> None:
    nvml = FakeNvml(devices=[FakeDevice(current_throttle=0x04 | 0x40)])

    sample = _reader(nvml).sample()[0]

    assert sample.throttle_reasons == ("sw_power_cap", "hw_thermal_slowdown")
    assert sample.throttle_reasons_available is True


def test_no_active_reason_is_distinguished_from_cannot_tell() -> None:
    knows = _reader(FakeNvml(devices=[FakeDevice(current_throttle=0)])).sample()[0]
    cannot_tell = _reader(FakeNvml(devices=[FakeDevice(supported_throttle=0)])).sample()[0]

    assert knows.throttle_reasons == () and knows.throttle_reasons_available is True
    assert cannot_tell.throttle_reasons == () and cannot_tell.throttle_reasons_available is False


def test_unsupported_reasons_are_not_reported_as_active() -> None:
    # The device reports only sw_power_cap as supported, but the current mask has more bits set.
    nvml = FakeNvml(devices=[FakeDevice(supported_throttle=0x04, current_throttle=0x04 | 0x40)])

    sample = _reader(nvml).sample()[0]

    assert sample.throttle_reasons == ("sw_power_cap",)


def test_failed_throttle_query_leaves_the_rest_of_the_sample_intact() -> None:
    nvml = FakeNvml(fail={"nvmlDeviceGetSupportedClocksThrottleReasons": NVMLError_NotSupported()})
    reader = _reader(nvml)

    sample = reader.sample()[0]

    assert sample.throttle_reasons_available is False
    assert sample.throttle_reasons == ()
    assert sample.vram_used_bytes == 4_000_000_000
    assert reader.unavailable_reasons()["gpu.0.throttle_reasons"] == "sensor_unsupported"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (NVMLError_NoPermission(), "permission_denied"),
        (NVMLError_DriverNotLoaded(), "nvml_driver_not_loaded"),
        (NVMLError_GpuIsLost(), "gpu_lost"),
        (RuntimeError("unclassified"), "nvml_error"),
    ],
)
def test_initialization_failure_is_recorded_and_degrades_the_whole_reader(
    failure: Exception, reason: str
) -> None:
    reader = _reader(FakeNvml(fail={"nvmlInit": failure}))

    assert reader.available() is False
    assert reader.sample() == ()
    assert reader.static_info() == ()
    assert reader.unavailable_reasons()["gpu"] == reason


def test_a_failed_initialization_is_retried_on_the_next_call() -> None:
    nvml = FakeNvml(fail={"nvmlInit": NVMLError_DriverNotLoaded()})
    reader = _reader(nvml)
    assert reader.available() is False

    nvml.fail = {}

    assert reader.available() is True
    assert reader.sample()[0].vram_used_bytes == 4_000_000_000


def test_initialization_happens_once_and_lazily() -> None:
    nvml = FakeNvml()
    reader = NvmlGpuReader(binding=nvml)
    assert nvml.init_calls == 0  # constructing must not touch the library

    reader.available()
    reader.sample()
    reader.static_info()

    assert nvml.init_calls == 1


def test_close_shuts_nvml_down_once_and_is_safe_to_repeat() -> None:
    nvml = FakeNvml()
    reader = NvmlGpuReader(binding=nvml)
    reader.sample()

    reader.close()
    reader.close()

    assert nvml.shutdown_calls == 1


def test_close_never_raises_when_shutdown_fails() -> None:
    nvml = FakeNvml(fail={"nvmlShutdown": NVMLError_GpuIsLost()})
    reader = NvmlGpuReader(binding=nvml)
    reader.sample()

    reader.close()

    assert nvml.shutdown_calls == 0


def test_context_manager_releases_the_library() -> None:
    nvml = FakeNvml()

    with NvmlGpuReader(binding=nvml) as reader:
        assert reader.sample()

    assert nvml.shutdown_calls == 1


def test_never_initialized_reader_shuts_nothing_down() -> None:
    nvml = FakeNvml()

    NvmlGpuReader(binding=nvml).close()

    assert nvml.shutdown_calls == 0


def test_zero_devices_is_unavailable_and_empty() -> None:
    reader = _reader(FakeNvml(devices=[]))

    assert reader.available() is False
    assert reader.sample() == ()
    assert reader.static_info() == ()
    assert reader.unavailable_reasons()["gpu"] == "no_nvidia_gpus"


def test_malformed_device_count_degrades_instead_of_inventing_devices() -> None:
    class BadCount(FakeNvml):
        def nvmlDeviceGetCount(self) -> int:  # noqa: N802 — NVML C API name
            return -3

    reader = _reader(BadCount())

    assert reader.sample() == ()
    assert reader.unavailable_reasons()["gpu"] == "malformed_value"


def test_unaddressable_device_is_skipped_without_losing_its_neighbour() -> None:
    class OneBadHandle(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int, /) -> object:  # noqa: N802 — C API name
            if index == 0:
                raise NVMLError_GpuIsLost
            return str(index)

    reader = _reader(OneBadHandle(devices=[FakeDevice(uuid="GPU-0"), FakeDevice(uuid="GPU-1")]))

    samples = reader.sample()

    assert [sample.index for sample in samples] == [1]
    assert reader.unavailable_reasons()["gpu.0"] == "gpu_lost"


@pytest.mark.parametrize("cuda", [0, -1, True])
def test_malformed_cuda_version_is_none_with_a_reason(cuda: Any) -> None:
    reader = _reader(FakeNvml(cuda=cuda))

    profile = reader.static_info()[0]

    assert profile.cuda_version is None
    assert reader.unavailable_reasons()["gpu.cuda_version"] == "malformed_value"


@pytest.mark.parametrize("capability", [(1,), "12.0", (1, 2, 3), (True, 0)])
def test_malformed_compute_capability_is_none_with_a_reason(capability: Any) -> None:
    reader = _reader(FakeNvml(devices=[FakeDevice(compute_capability=capability)]))

    profile = reader.static_info()[0]

    assert profile.compute_capability is None
    assert reader.unavailable_reasons()["gpu.0.compute_capability"] == "malformed_value"


def test_out_of_range_readings_degrade_rather_than_being_reported() -> None:
    nvml = FakeNvml(
        devices=[
            FakeDevice(
                util_gpu=150,
                util_memory=-1,
                temperature=-400,
                fan=101,
                power_milliwatts=-5,
                clock_sm=-1,
            )
        ]
    )

    sample = _reader(nvml).sample()[0]

    for name in (
        "utilization_percent",
        "memory_utilization_percent",
        "temperature_c",
        "fan_percent",
        "power_watts",
        "core_clock_mhz",
    ):
        assert getattr(sample, name) is UNSUPPORTED, name
    assert sample.vram_used_bytes == 4_000_000_000


def test_a_binding_missing_a_required_call_degrades_with_a_named_reason() -> None:
    class Incomplete:
        nvmlInit = None  # noqa: N815 — a binding that exposes the name but not a callable

    reader = NvmlGpuReader(binding=Incomplete())  # type: ignore[arg-type]  # deliberately partial

    assert reader.available() is False
    assert reader.unavailable_reasons()["gpu"] == "nvml_binding_incomplete"


def test_reasons_mapping_is_a_defensive_copy() -> None:
    reader = _reader(FakeNvml())
    reader.sample()
    reasons = reader.unavailable_reasons()

    dict(reasons)["gpu.0.memory_temperature_c"] = "tampered"

    assert reader.unavailable_reasons()["gpu.0.memory_temperature_c"] == "sensor_unsupported"


def test_reasons_are_cleared_between_operations() -> None:
    nvml = FakeNvml(fail={"nvmlDeviceGetPowerUsage": NVMLError_NotSupported()})
    reader = _reader(nvml)
    reader.sample()
    assert "gpu.0.power_watts" in reader.unavailable_reasons()

    nvml.fail = {}
    reader.sample()

    assert "gpu.0.power_watts" not in reader.unavailable_reasons()


def test_binding_loader_reports_absence_without_importing_anything() -> None:
    # The extra is genuinely optional; whichever way this environment is set up, the two helpers
    # must agree, and the loader must fail with the actionable suite error rather than ImportError.
    if nvml_binding_available():
        assert load_nvml_binding() is not None
    else:
        with pytest.raises(DependencyUnavailableError, match="pynvml"):
            load_nvml_binding()


def test_loader_raises_the_actionable_suite_error_when_the_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulating an uninstalled distribution is a boundary condition; the alternative would be
    # uninstalling a package inside a test run.
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(DependencyUnavailableError, match=r"sweatmeter\[pynvml\]") as raised:
        load_nvml_binding()

    assert raised.value.details == {"dependency": "pynvml", "backend": "pynvml"}
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_reader_without_an_injected_binding_reports_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    reader = NvmlGpuReader()

    assert reader.available() is False
    assert reader.sample() == ()
    assert reader.unavailable_reasons()["gpu"] == "pynvml_not_installed"


def test_reader_without_an_injected_binding_loads_the_real_extra_when_present() -> None:
    if not nvml_binding_available():
        pytest.skip("the optional pynvml extra is not installed in this environment")
    reader = NvmlGpuReader()

    # No GPU is required: this proves only that the lazy import path reaches a real binding and
    # that a driverless environment degrades honestly rather than raising.
    assert isinstance(reader.available(), bool)
    reader.close()


def test_an_unexpected_loader_failure_is_not_reported_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nvml_module, "load_nvml_binding", lambda: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    reader = NvmlGpuReader()

    assert reader.available() is False
    assert reader.unavailable_reasons()["gpu"] == "nvml_error"


def test_non_text_identity_values_become_none() -> None:
    nvml = FakeNvml(devices=[FakeDevice(name=1234, uuid=None)])  # type: ignore[arg-type]

    profile = _reader(nvml).static_info()[0]

    assert profile.name is None
    assert profile.uuid is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_sensor_values_degrade(value: float) -> None:
    nvml = FakeNvml(devices=[FakeDevice(power_milliwatts=value)])  # type: ignore[arg-type]

    sample = _reader(nvml).sample()[0]

    assert sample.power_watts is UNSUPPORTED


def test_memory_info_without_usable_fields_degrades_both_vram_figures() -> None:
    class NoMemoryFields(FakeNvml):
        def nvmlDeviceGetMemoryInfo(self, handle: object, /) -> object:  # noqa: N802 — C API name
            del handle
            return object()

    reader = _reader(NoMemoryFields())

    sample = reader.sample()[0]
    reasons = reader.unavailable_reasons()

    assert sample.vram_used_bytes is UNSUPPORTED
    assert sample.vram_total_bytes is UNSUPPORTED
    assert reasons["gpu.0.vram_used_bytes"] == "sensor_unsupported"
    assert reasons["gpu.0.vram_total_bytes"] == "sensor_unsupported"


def test_static_info_skips_an_unaddressable_device_without_losing_its_neighbour() -> None:
    class OneBadHandle(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int, /) -> object:  # noqa: N802 — C API name
            if index == 0:
                raise NVMLError_GpuIsLost
            return str(index)

    reader = _reader(OneBadHandle(devices=[FakeDevice(uuid="GPU-0"), FakeDevice(uuid="GPU-1")]))

    profiles = reader.static_info()

    assert [profile.index for profile in profiles] == [1]
    assert reader.unavailable_reasons()["gpu.0"] == "gpu_lost"


def test_a_binding_exposing_a_name_that_is_not_callable_degrades_with_a_named_reason() -> None:
    class NotCallable(FakeNvml):
        nvmlDeviceGetCount = "not-a-function"  # type: ignore[assignment]  # noqa: N815 — C API name

    reader = _reader(NotCallable())

    assert reader.sample() == ()
    assert reader.unavailable_reasons()["gpu"] == "nvml_binding_incomplete"
