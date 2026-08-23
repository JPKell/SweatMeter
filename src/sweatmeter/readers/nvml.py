"""NVIDIA telemetry through the in-process NVML binding supplied by the ``pynvml`` extra.

This backend answers the same ``GpuReader`` questions as :mod:`sweatmeter.readers.nvidia` without
starting a process per sample. Two consequences follow, and both matter to the contract rather than
only to speed. Each metric is its own NVML call, so an unsupported sensor degrades exactly that
field instead of risking the command that carries its neighbours. And the driver reports which
throttle reasons it is *able* to report, so ``throttle_reasons_available`` is answered from the
device rather than inferred from parseable text.

``pynvml`` is an optional extra ([ADR-0021](../../docs/adr/0021-telemetry-collection-strategy.md)
§7), so this module imports it lazily and never at module scope: importing SweatMeter must keep
working with no NVIDIA tooling installed at all.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Protocol, cast, runtime_checkable

from baseaicore import (
    UNSUPPORTED,
    DependencyUnavailableError,
    GpuProfile,
    GpuVendor,
    Measurement,
)

from sweatmeter.safe import _safe_call
from sweatmeter.types import GpuSample

__all__ = ["NvmlBinding", "NvmlGpuReader", "load_nvml_binding", "nvml_binding_available"]

_LOGGER = logging.getLogger(__name__)
_BINDING_MODULE = "pynvml"
_MILLI = 1000.0
_PERCENT_MAXIMUM = 100.0
_MINIMUM_TEMPERATURE_C = -273.15
_CUDA_MAJOR_DIVISOR = 1000
_CUDA_MINOR_DIVISOR = 10
_COMPUTE_CAPABILITY_PARTS = 2

# Bit values from NVML's `nvmlClocksThrottleReason*` constants. They are hard-coded rather than
# read back from the binding so that a binding which omits one cannot silently shift every other
# reason, and so the names match `sweatmeter.readers.nvidia` exactly: the two backends must be
# interchangeable for `TelemetryWindow`'s throttle heuristic, which keys on these strings.
_THROTTLE_BITS: tuple[tuple[str, int], ...] = (
    ("gpu_idle", 0x0000000000000001),
    ("applications_clocks_setting", 0x0000000000000002),
    ("sw_power_cap", 0x0000000000000004),
    ("hw_slowdown", 0x0000000000000008),
    ("sync_boost", 0x0000000000000010),
    ("sw_thermal_slowdown", 0x0000000000000020),
    ("hw_thermal_slowdown", 0x0000000000000040),
    ("hw_power_brake_slowdown", 0x0000000000000080),
)

# NVML raises one exception class per status code. Matching on the class *name* keeps this mapping
# free of any import from the optional extra, so the reasons stay available even when the binding
# is a test double or a different NVML wrapper.
_ERROR_REASONS: Mapping[str, str] = {
    "NVMLError_NotSupported": "sensor_unsupported",
    "NVMLError_InvalidArgument": "sensor_unsupported",
    "NVMLError_FunctionNotFound": "sensor_unsupported",
    "NVMLError_NoPermission": "permission_denied",
    "NVMLError_LibraryNotFound": "nvml_library_not_found",
    "NVMLError_DriverNotLoaded": "nvml_driver_not_loaded",
    "NVMLError_Uninitialized": "nvml_not_initialized",
    "NVMLError_GpuIsLost": "gpu_lost",
    "NVMLError_Timeout": "nvml_timeout",
}
_DEFAULT_REASON = "nvml_error"


@runtime_checkable
class NvmlBinding(Protocol):
    """The subset of an NVML binding that SweatMeter calls.

    ``pynvml`` satisfies this structurally, and so does any wrapper exposing the same C-derived
    names. Only the read operations below are ever invoked: this package never sets a clock, a fan
    curve or a power limit. Tests supply a double, so no test needs the library or a GPU.

    The mixed-case names mirror the NVML C API deliberately; renaming them here would break
    structural matching against the real binding.
    """

    NVML_TEMPERATURE_GPU: int
    NVML_CLOCK_SM: int
    NVML_CLOCK_MEM: int

    def nvmlInit(self) -> None:  # noqa: N802 — NVML C API name
        """Initialize the NVML library."""
        ...

    def nvmlShutdown(self) -> None:  # noqa: N802 — NVML C API name
        """Release the NVML library."""
        ...

    def nvmlSystemGetDriverVersion(self) -> str | bytes:  # noqa: N802 — NVML C API name
        """Return the installed display-driver version."""
        ...

    def nvmlSystemGetCudaDriverVersion(self) -> int:  # noqa: N802 — NVML C API name
        """Return the CUDA driver version as ``major * 1000 + minor * 10``."""
        ...

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802 — NVML C API name
        """Return the number of visible NVIDIA devices."""
        ...

    def nvmlDeviceGetHandleByIndex(self, index: int, /) -> object:  # noqa: N802 — NVML C API name
        """Return an opaque device handle for one index."""
        ...

    def nvmlDeviceGetName(self, handle: object, /) -> str | bytes:  # noqa: N802 — NVML C API name
        """Return one device's marketing name."""
        ...

    def nvmlDeviceGetUUID(self, handle: object, /) -> str | bytes:  # noqa: N802 — NVML C API name
        """Return one device's stable UUID."""
        ...

    def nvmlDeviceGetMemoryInfo(self, handle: object, /) -> object:  # noqa: N802 — NVML C API name
        """Return an object exposing ``total`` and ``used`` VRAM in bytes."""
        ...

    def nvmlDeviceGetUtilizationRates(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> object:
        """Return an object exposing ``gpu`` and ``memory`` utilization percentages."""
        ...

    def nvmlDeviceGetTemperature(  # noqa: N802 — NVML C API name
        self, handle: object, sensor: int, /
    ) -> int:
        """Return one sensor's temperature in degrees Celsius."""
        ...

    def nvmlDeviceGetPowerUsage(self, handle: object, /) -> int:  # noqa: N802 — NVML C API name
        """Return instantaneous board power draw in milliwatts."""
        ...

    def nvmlDeviceGetEnforcedPowerLimit(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        """Return the enforced power limit in milliwatts."""
        ...

    def nvmlDeviceGetFanSpeed(self, handle: object, /) -> int:  # noqa: N802 — NVML C API name
        """Return fan speed as a percentage of its maximum."""
        ...

    def nvmlDeviceGetClockInfo(  # noqa: N802 — NVML C API name
        self, handle: object, clock: int, /
    ) -> int:
        """Return one clock domain's current frequency in MHz."""
        ...

    def nvmlDeviceGetCudaComputeCapability(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> tuple[int, int]:
        """Return one device's ``(major, minor)`` compute capability."""
        ...

    def nvmlDeviceGetCurrentClocksThrottleReasons(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        """Return the bitmask of currently active throttle reasons."""
        ...

    def nvmlDeviceGetSupportedClocksThrottleReasons(  # noqa: N802 — NVML C API name
        self, handle: object, /
    ) -> int:
        """Return the bitmask of throttle reasons this device can report at all."""
        ...


def nvml_binding_available() -> bool:
    """Report whether the optional NVML binding can be imported, without importing it.

    Returns:
        True when the ``pynvml`` extra is installed. This inspects the import system only; it
        neither loads the library nor touches a device, so it is safe in a factory.
    """
    found = _safe_call(lambda: importlib.util.find_spec(_BINDING_MODULE), None)[0]
    return found is not None


def load_nvml_binding() -> NvmlBinding:
    """Import the optional NVML binding.

    Returns:
        The imported module, which satisfies :class:`NvmlBinding` structurally.

    Raises:
        DependencyUnavailableError: If the ``pynvml`` extra is not installed. The message names the
            extra to install, because this is a packaging gap rather than a missing GPU.
    """
    module, failure = _safe_call(lambda: importlib.import_module(_BINDING_MODULE), None)
    if module is None:
        raise DependencyUnavailableError(
            "The NVML backend needs the optional pynvml extra; install 'sweatmeter[pynvml]' or "
            "select the nvidia-smi backend instead.",
            details={"dependency": _BINDING_MODULE, "backend": "pynvml"},
        ) from failure
    return cast(NvmlBinding, module)


def _text(value: object) -> str | None:
    """Normalize NVML text, which older bindings return as bytes, to non-empty text or ``None``."""
    if isinstance(value, bytes):
        raw = value
        decoded, _failure = _safe_call(lambda: raw.decode("utf-8", errors="replace"), None)
        value = decoded
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _number(
    value: object,
    *,
    scale: float = 1.0,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> Measurement:
    """Return a finite, in-range, normalized measurement or the unsupported sentinel."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNSUPPORTED
    if not math.isfinite(value):
        return UNSUPPORTED
    scaled = value / scale if scale != 1.0 else value
    if not math.isfinite(scaled) or scaled < minimum:
        return UNSUPPORTED
    if maximum is not None and scaled > maximum:
        return UNSUPPORTED
    return scaled


class NvmlGpuReader:
    """Read per-device NVIDIA telemetry in-process, without fabricating unavailable values.

    The library is initialized lazily on first use and kept initialized, because the cost this
    backend exists to avoid is per-call setup. A failed initialization degrades the whole reader
    for that call only and is retried on the next one, matching
    :class:`~sweatmeter.readers.nvidia.NvidiaSmiReader`. Every metric is read independently, so one
    unsupported sensor cannot remove another.

    ``unavailable_reasons()`` describes degradation from the most recent public operation. Reader
    instances retain only that mapping plus the initialization flag, and are intended to be owned
    by one collector; concurrent use of one instance is not supported.

    Call :meth:`close`, or use the reader as a context manager, to release NVML. A process that
    simply exits does not need to.

    Args:
        binding: NVML binding to call. Defaults to importing the optional ``pynvml`` extra on
            first use, so construction neither imports the library nor touches a device.
    """

    def __init__(self, *, binding: NvmlBinding | None = None) -> None:
        """Configure the NVML boundary without loading the library or probing hardware."""
        self._binding = binding
        self._initialized = False
        self._reasons: dict[str, str] = {}

    def __enter__(self) -> NvmlGpuReader:
        """Return this reader; NVML is initialized on first use, not on entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release NVML on every context-manager exit path."""
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        """Release NVML if this reader initialized it, and never raise while doing so."""
        binding, initialized, self._initialized = self._binding, self._initialized, False
        if binding is not None and initialized:
            self._attempt(binding.nvmlShutdown, reason_key="gpu")

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return a copy of degradation reasons from the most recent public operation."""
        return dict(sorted(self._reasons.items()))

    def available(self) -> bool:
        """Return whether at least one NVIDIA device can be queried through NVML now.

        The result is a live probe. An installed binding with no working driver or visible device
        is unavailable, and a later call retries it.
        """
        self._reasons.clear()
        binding = self._ready()
        if binding is None:
            return False
        count = self._device_count(binding)
        if count <= 0:
            self._reasons.setdefault("gpu", "no_nvidia_gpus")
            return False
        return True

    def sample(self) -> tuple[GpuSample, ...]:
        """Return one live telemetry sample per visible NVIDIA device.

        Each sensor is read on its own, so a device that cannot report power still reports VRAM,
        and a missing sensor becomes ``UNSUPPORTED`` with a recorded reason rather than zero.
        """
        self._reasons.clear()
        binding = self._ready()
        if binding is None:
            return ()
        samples: list[GpuSample] = []
        for index in range(self._device_count(binding)):
            handle = self._handle(binding, index)
            if handle is None:
                continue
            samples.append(self._sample_device(binding, handle, index=index))
        if not samples:
            self._reasons.setdefault("gpu", "no_nvidia_gpus")
        return tuple(samples)

    def static_info(self) -> tuple[GpuProfile, ...]:
        """Return static identity and capacity information for every visible NVIDIA device.

        Driver and CUDA versions are system-wide NVML queries; either may fail while the per-device
        profiles remain available.
        """
        self._reasons.clear()
        binding = self._ready()
        if binding is None:
            return ()
        driver_version = _text(
            self._attempt(binding.nvmlSystemGetDriverVersion, reason_key="gpu.driver_version")
        )
        cuda_version = self._cuda_version(binding)
        profiles: list[GpuProfile] = []
        for index in range(self._device_count(binding)):
            handle = self._handle(binding, index)
            if handle is None:
                continue
            profiles.append(
                GpuProfile(
                    index=index,
                    name=_text(
                        self._attempt(
                            lambda h=handle: binding.nvmlDeviceGetName(h),
                            reason_key=f"gpu.{index}.name",
                        )
                    ),
                    uuid=_text(
                        self._attempt(
                            lambda h=handle: binding.nvmlDeviceGetUUID(h),
                            reason_key=f"gpu.{index}.uuid",
                        )
                    ),
                    vram_total_bytes=self._memory(binding, handle, index=index)[1],
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    compute_capability=self._compute_capability(binding, handle, index=index),
                    vendor=GpuVendor.NVIDIA,
                )
            )
        if not profiles:
            self._reasons.setdefault("gpu", "no_nvidia_gpus")
        return tuple(profiles)

    def _attempt(self, operation: object, *, reason_key: str) -> object:
        """Call one NVML operation, recording a machine-readable reason when it fails."""
        if not callable(operation):
            self._reasons[reason_key] = "nvml_binding_incomplete"
            return None
        value, failure = _safe_call(operation, None)
        if failure is not None:
            reason = _ERROR_REASONS.get(type(failure).__name__, _DEFAULT_REASON)
            self._reasons[reason_key] = reason
            _LOGGER.debug(
                "telemetry.nvml.call_failed",
                extra={"operation": reason_key, "reason": reason},
            )
            return None
        return value

    def _ready(self) -> NvmlBinding | None:
        """Return an initialized binding, recording why one is unavailable."""
        if self._binding is None:
            binding, failure = _safe_call(load_nvml_binding, None)
            if binding is None:
                self._reasons["gpu"] = (
                    "pynvml_not_installed"
                    if isinstance(failure, DependencyUnavailableError)
                    else _DEFAULT_REASON
                )
                return None
            self._binding = binding
        if self._initialized:
            return self._binding
        initializer = getattr(self._binding, "nvmlInit", None)
        if not callable(initializer):
            self._reasons["gpu"] = "nvml_binding_incomplete"
            return None
        _value, failure = _safe_call(initializer, None)
        if failure is not None:
            self._reasons["gpu"] = _ERROR_REASONS.get(type(failure).__name__, _DEFAULT_REASON)
            _LOGGER.debug("telemetry.nvml.init_failed", extra={"reason": self._reasons["gpu"]})
            return None
        self._initialized = True
        return self._binding

    def _device_count(self, binding: NvmlBinding) -> int:
        """Return the visible device count, degrading a failed or nonsensical answer to zero."""
        count = self._attempt(binding.nvmlDeviceGetCount, reason_key="gpu")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            self._reasons.setdefault("gpu", "malformed_value")
            return 0
        return count

    def _handle(self, binding: NvmlBinding, index: int) -> object | None:
        """Return one device handle, or ``None`` when that device cannot be addressed."""
        return self._attempt(
            lambda: binding.nvmlDeviceGetHandleByIndex(index),
            reason_key=f"gpu.{index}",
        )

    def _memory(
        self, binding: NvmlBinding, handle: object, *, index: int
    ) -> tuple[Measurement, Measurement]:
        """Return one device's ``(used, total)`` VRAM in bytes."""
        info = self._attempt(
            lambda: binding.nvmlDeviceGetMemoryInfo(handle),
            reason_key=f"gpu.{index}.vram_used_bytes",
        )
        used = _number(getattr(info, "used", None))
        total = _number(getattr(info, "total", None))
        if used is UNSUPPORTED:
            self._reasons.setdefault(f"gpu.{index}.vram_used_bytes", "sensor_unsupported")
        if total is UNSUPPORTED:
            self._reasons.setdefault(f"gpu.{index}.vram_total_bytes", "sensor_unsupported")
        return (used, total)

    def _cuda_version(self, binding: NvmlBinding) -> str | None:
        """Return the CUDA driver version as ``major.minor``, or ``None`` when unavailable."""
        raw = self._attempt(binding.nvmlSystemGetCudaDriverVersion, reason_key="gpu.cuda_version")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            self._reasons.setdefault("gpu.cuda_version", "malformed_value")
            return None
        return f"{raw // _CUDA_MAJOR_DIVISOR}.{(raw % _CUDA_MAJOR_DIVISOR) // _CUDA_MINOR_DIVISOR}"

    def _compute_capability(
        self, binding: NvmlBinding, handle: object, *, index: int
    ) -> str | None:
        """Return one device's compute capability as ``major.minor``."""
        raw = self._attempt(
            lambda: binding.nvmlDeviceGetCudaComputeCapability(handle),
            reason_key=f"gpu.{index}.compute_capability",
        )
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != _COMPUTE_CAPABILITY_PARTS
        ):
            self._reasons.setdefault(f"gpu.{index}.compute_capability", "malformed_value")
            return None
        major, minor = raw
        if any(isinstance(part, bool) or not isinstance(part, int) for part in (major, minor)):
            self._reasons[f"gpu.{index}.compute_capability"] = "malformed_value"
            return None
        return f"{major}.{minor}"

    def _throttle_reasons(
        self, binding: NvmlBinding, handle: object, *, index: int
    ) -> tuple[tuple[str, ...], bool]:
        """Return active throttle reasons and whether "none active" is knowable for this device.

        NVML reports which reasons a device is *able* to report, so unlike the text backend this
        distinction comes from the driver rather than from what happened to parse.
        """
        reason_key = f"gpu.{index}.throttle_reasons"
        supported = self._attempt(
            lambda: binding.nvmlDeviceGetSupportedClocksThrottleReasons(handle),
            reason_key=reason_key,
        )
        current = self._attempt(
            lambda: binding.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
            reason_key=reason_key,
        )
        if (
            isinstance(supported, bool)
            or not isinstance(supported, int)
            or isinstance(current, bool)
            or not isinstance(current, int)
            or supported <= 0
            or current < 0
        ):
            self._reasons.setdefault(reason_key, "sensor_unsupported")
            return ((), False)
        active = tuple(name for name, bit in _THROTTLE_BITS if supported & bit and current & bit)
        return (active, True)

    def _sample_device(self, binding: NvmlBinding, handle: object, *, index: int) -> GpuSample:
        """Build one normalized live sample from independent NVML reads."""
        used_bytes, total_bytes = self._memory(binding, handle, index=index)
        utilization = self._attempt(
            lambda: binding.nvmlDeviceGetUtilizationRates(handle),
            reason_key=f"gpu.{index}.utilization_percent",
        )
        reasons, reasons_available = self._throttle_reasons(binding, handle, index=index)
        sample = GpuSample(
            index=index,
            uuid=_text(
                self._attempt(
                    lambda: binding.nvmlDeviceGetUUID(handle), reason_key=f"gpu.{index}.uuid"
                )
            ),
            utilization_percent=_number(
                getattr(utilization, "gpu", None), maximum=_PERCENT_MAXIMUM
            ),
            memory_utilization_percent=_number(
                getattr(utilization, "memory", None), maximum=_PERCENT_MAXIMUM
            ),
            vram_used_bytes=used_bytes,
            vram_total_bytes=total_bytes,
            temperature_c=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetTemperature(handle, binding.NVML_TEMPERATURE_GPU),
                    reason_key=f"gpu.{index}.temperature_c",
                ),
                minimum=_MINIMUM_TEMPERATURE_C,
            ),
            # NVML exposes no portable memory-junction sensor, so this field is honestly
            # unsupported on this backend rather than approximated from the core temperature.
            memory_temperature_c=UNSUPPORTED,
            power_watts=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetPowerUsage(handle),
                    reason_key=f"gpu.{index}.power_watts",
                ),
                scale=_MILLI,
            ),
            power_limit_watts=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetEnforcedPowerLimit(handle),
                    reason_key=f"gpu.{index}.power_limit_watts",
                ),
                scale=_MILLI,
            ),
            fan_percent=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetFanSpeed(handle),
                    reason_key=f"gpu.{index}.fan_percent",
                ),
                maximum=_PERCENT_MAXIMUM,
            ),
            core_clock_mhz=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetClockInfo(handle, binding.NVML_CLOCK_SM),
                    reason_key=f"gpu.{index}.core_clock_mhz",
                )
            ),
            memory_clock_mhz=_number(
                self._attempt(
                    lambda: binding.nvmlDeviceGetClockInfo(handle, binding.NVML_CLOCK_MEM),
                    reason_key=f"gpu.{index}.memory_clock_mhz",
                )
            ),
            throttle_reasons=reasons,
            throttle_reasons_available=reasons_available,
        )
        self._reasons.setdefault(f"gpu.{index}.memory_temperature_c", "sensor_unsupported")
        for field in (
            "utilization_percent",
            "memory_utilization_percent",
            "temperature_c",
            "power_watts",
            "power_limit_watts",
            "fan_percent",
            "core_clock_mhz",
            "memory_clock_mhz",
        ):
            if getattr(sample, field) is UNSUPPORTED:
                self._reasons.setdefault(f"gpu.{index}.{field}", "sensor_unsupported")
        return sample
