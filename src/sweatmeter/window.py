"""Per-device aggregates and telemetry-derived energy and throttling estimates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from baseaicore import UNSUPPORTED, Measurement, ValidationError, is_supported

from sweatmeter.types import GpuSample, TelemetrySnapshot

__all__ = ["TelemetryWindow", "ThrottleState", "ThrottleVerdict", "WindowMetric"]

_CLOCK_DROP_FRACTION = 0.15
_HIGH_TEMPERATURE_C = 80.0
_LOW_UTILIZATION_PERCENT = 50.0
_POWER_LIMIT_FRACTION = 0.95
_MIN_CLOCK_SAMPLES = 2
_ACTIVE_THROTTLE_EXCLUSIONS = frozenset({"gpu_idle"})


class WindowMetric(StrEnum):
    """Derived metric names accepted by :meth:`TelemetryWindow.supported_sample_count`."""

    PEAK_VRAM_BYTES = "peak_vram_bytes"
    MEAN_POWER_WATTS = "mean_power_watts"
    ENERGY_JOULES = "energy_joules"
    MAX_TEMPERATURE_C = "max_temperature_c"
    SUSPECTED_THROTTLING = "suspected_throttling"


class ThrottleState(StrEnum):
    """Three-state outcome of the conservative throttling heuristic."""

    SUSPECTED = "suspected"
    NOT_SUSPECTED = "not_suspected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ThrottleVerdict:
    """Explain a per-device heuristic throttling assessment.

    This is deliberately not a boolean: a missing clock sensor is unknown, not evidence of no
    throttling. Confidence expresses the strength of the available telemetry rather than a
    calibrated hardware probability.

    Attributes:
        gpu_index: Device index whose samples were examined.
        verdict: Suspected, not suspected, or unknown.
        reason: Stable machine-readable explanation for the verdict.
        confidence: Heuristic evidence strength from 0.0 through 1.0.
        supported_samples: Number of device samples that supplied the decisive evidence.

    Raises:
        ValidationError: If the device index, confidence, count, or reason is invalid.
    """

    gpu_index: int
    verdict: ThrottleState
    reason: str
    confidence: float
    supported_samples: int

    def __post_init__(self) -> None:
        """Validate the public verdict invariants."""
        if (
            isinstance(self.gpu_index, bool)
            or not isinstance(self.gpu_index, int)
            or self.gpu_index < 0
        ):
            raise ValidationError(
                f"ThrottleVerdict.gpu_index must be a non-negative integer; got "
                f"{self.gpu_index!r}.",
                details={"field": "gpu_index", "value": self.gpu_index},
            )
        if not isinstance(self.verdict, ThrottleState):
            raise ValidationError(
                f"ThrottleVerdict.verdict must be a ThrottleState; got {self.verdict!r}.",
                details={"field": "verdict", "value": str(self.verdict)},
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValidationError(
                "ThrottleVerdict.reason must be a non-empty string.",
                details={"field": "reason", "value": str(self.reason)},
            )
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValidationError(
                f"ThrottleVerdict.confidence must be between 0 and 1; got {self.confidence!r}.",
                details={"field": "confidence", "value": self.confidence},
            )
        if (
            isinstance(self.supported_samples, bool)
            or not isinstance(self.supported_samples, int)
            or self.supported_samples < 0
        ):
            raise ValidationError(
                "ThrottleVerdict.supported_samples must be a non-negative integer; got "
                f"{self.supported_samples!r}.",
                details={"field": "supported_samples", "value": self.supported_samples},
            )

    @property
    def suspected(self) -> bool | None:
        """Return a boolean assessment, or ``None`` when telemetry cannot decide."""
        if self.verdict is ThrottleState.UNKNOWN:
            return None
        return self.verdict is ThrottleState.SUSPECTED


def _validate_gpu_index(gpu_index: object) -> int:
    """Return a non-negative GPU index or raise the suite validation error."""
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValidationError(
            f"gpu_index must be a non-negative integer; got {gpu_index!r}.",
            details={"field": "gpu_index", "value": gpu_index},
        )
    return gpu_index


class TelemetryWindow:
    """Aggregate an immutable, chronological sequence of telemetry snapshots.

    All GPU figures are per device and require an explicit or default ``gpu_index``; this package
    never sums or averages devices into a machine-wide number (ADR-0027). Unsupported values are
    excluded rather than treated as zero. :meth:`supported_sample_count` reports how much evidence
    each result used.

    Energy is a telemetry-derived estimate, not hardware instrumentation. It uses a left-rectangle
    sum: each supported power sample is multiplied by the actual time until the next snapshot.
    The final sample has no following duration and contributes no energy. Missing power intervals
    are skipped rather than assigned zero.

    Args:
        samples: Snapshot series. It is copied and stably sorted by timestamp, so caller mutation or
            out-of-order persistence cannot produce negative integration intervals.

    Raises:
        ValidationError: If an element is not a ``TelemetrySnapshot``.
    """

    def __init__(self, samples: Sequence[TelemetrySnapshot]) -> None:
        """Copy and chronologically normalize the input series."""
        copied = tuple(samples)
        for position, sample in enumerate(copied):
            if not isinstance(sample, TelemetrySnapshot):
                raise ValidationError(
                    f"TelemetryWindow.samples[{position}] must be a TelemetrySnapshot; got "
                    f"{type(sample).__name__}.",
                    details={"field": f"samples.{position}"},
                )
        self._samples = tuple(sorted(copied, key=lambda sample: sample.timestamp))

    def sample_count(self) -> int:
        """Return the number of host snapshots in this window."""
        return len(self._samples)

    def peak_vram_bytes(self, gpu_index: int = 0) -> Measurement:
        """Return one device's maximum supported used-VRAM reading.

        Args:
            gpu_index: Non-negative device index; devices are never aggregated.

        Returns:
            Peak bytes, or ``UNSUPPORTED`` when no matching sample reports used VRAM.
        """
        values = self._measurements(gpu_index, "vram_used_bytes")
        return max(values) if values else UNSUPPORTED

    def mean_power_watts(self, gpu_index: int = 0) -> Measurement:
        """Return one device's mean over supported power samples.

        Args:
            gpu_index: Non-negative device index; devices are never aggregated.

        Returns:
            Arithmetic mean watts, or ``UNSUPPORTED`` when no power sensor reading exists.
        """
        values = self._measurements(gpu_index, "power_watts")
        return sum(values) / len(values) if values else UNSUPPORTED

    def energy_joules(self, gpu_index: int = 0) -> Measurement:
        """Estimate one device's energy as ``sum(power_watts * actual_dt_seconds)``.

        This is explicitly a telemetry-derived estimate, never hardware instrumentation. Each
        supported power sample applies until the next real snapshot timestamp; the nominal sampler
        interval is never used.

        Args:
            gpu_index: Non-negative device index; devices are never summed.

        Returns:
            Estimated joules across supported intervals, or ``UNSUPPORTED`` if none can be
            integrated. A real measured zero-watt interval returns numeric zero.
        """
        terms = self._energy_terms(gpu_index)
        return sum(terms) if terms else UNSUPPORTED

    def max_temperature_c(self, gpu_index: int = 0) -> Measurement:
        """Return one device's maximum supported GPU-core temperature.

        Args:
            gpu_index: Non-negative device index; devices are never aggregated.

        Returns:
            Maximum degrees Celsius, or ``UNSUPPORTED`` when the sensor is unavailable throughout.
        """
        values = self._measurements(gpu_index, "temperature_c")
        return max(values) if values else UNSUPPORTED

    def suspected_throttling(self, gpu_index: int = 0) -> ThrottleVerdict:
        """Return a conservative, explained throttling heuristic for one device.

        Driver-reported active reasons are strongest. Otherwise a core-clock drop of at least 15%
        is correlated with temperature, power-limit proximity, or utilization. A drop at low
        utilization is classified as not suspected; missing clock data is unknown rather than
        false.

        Args:
            gpu_index: Non-negative device index; devices are never aggregated.

        Returns:
            Verdict with a stable reason, confidence, and evidence sample count.
        """
        index = _validate_gpu_index(gpu_index)
        samples = self._gpu_samples(index)
        reported = tuple(sample for sample in samples if sample.throttle_reasons_available)
        active = sorted(
            {
                reason
                for sample in reported
                for reason in sample.throttle_reasons
                if reason not in _ACTIVE_THROTTLE_EXCLUSIONS
            }
        )
        if active:
            return ThrottleVerdict(
                index,
                ThrottleState.SUSPECTED,
                f"driver_reported:{','.join(active)}",
                1.0,
                len(reported),
            )

        return self._clock_verdict(index, samples, reported)

    def supported_sample_count(self, metric: WindowMetric | str, gpu_index: int = 0) -> int:
        """Return the supported evidence count behind one per-device derived result.

        Args:
            metric: A ``WindowMetric`` or its exact string value.
            gpu_index: Non-negative device index; devices are never aggregated.

        Returns:
            Number of supported readings used. Energy counts supported interval-start samples, not
            the final power sample that has no following duration.

        Raises:
            ValidationError: If ``metric`` is unknown or ``gpu_index`` is invalid.
        """
        try:
            selected = WindowMetric(metric)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Unknown window metric {metric!r}; expected one of "
                f"{[item.value for item in WindowMetric]!r}.",
                details={"field": "metric", "value": str(metric)},
            ) from exc
        if selected is WindowMetric.PEAK_VRAM_BYTES:
            return len(self._measurements(gpu_index, "vram_used_bytes"))
        if selected is WindowMetric.MEAN_POWER_WATTS:
            return len(self._measurements(gpu_index, "power_watts"))
        if selected is WindowMetric.ENERGY_JOULES:
            return len(self._energy_terms(gpu_index))
        if selected is WindowMetric.MAX_TEMPERATURE_C:
            return len(self._measurements(gpu_index, "temperature_c"))
        return self.suspected_throttling(gpu_index).supported_samples

    def _gpu_samples(self, gpu_index: int) -> tuple[GpuSample, ...]:
        """Return one matching device sample from every snapshot that contains it."""
        index = _validate_gpu_index(gpu_index)
        matched: list[GpuSample] = []
        for snapshot in self._samples:
            sample = next((gpu for gpu in snapshot.gpus if gpu.index == index), None)
            if sample is not None:
                matched.append(sample)
        return tuple(matched)

    def _measurements(self, gpu_index: int, field: str) -> tuple[int | float, ...]:
        """Return supported numeric values for one device field."""
        values: list[int | float] = []
        for sample in self._gpu_samples(gpu_index):
            value = getattr(sample, field)
            if is_supported(value):
                values.append(value)
        return tuple(values)

    def _energy_terms(self, gpu_index: int) -> tuple[float, ...]:
        """Return supported left-rectangle power-by-time terms for one device."""
        index = _validate_gpu_index(gpu_index)
        terms: list[float] = []
        for current, following in pairwise(self._samples):
            elapsed_seconds = (following.timestamp - current.timestamp).total_seconds()
            if elapsed_seconds <= 0:
                continue
            gpu = next((sample for sample in current.gpus if sample.index == index), None)
            if gpu is not None and is_supported(gpu.power_watts):
                terms.append(float(gpu.power_watts) * elapsed_seconds)
        return tuple(terms)

    @staticmethod
    def _substantial_clock_drop(samples: tuple[GpuSample, ...]) -> GpuSample | None:
        """Return the first sample at least 15% below a previously observed clock maximum.

        Samples whose clock sensor is unavailable are skipped here rather than filtered by the
        caller, so an absent reading never stands in as a low clock.
        """
        maximum_clock: int | float | None = None
        for sample in samples:
            if not is_supported(sample.core_clock_mhz):
                continue
            clock = sample.core_clock_mhz
            if maximum_clock is None:
                maximum_clock = clock
                continue
            if clock <= maximum_clock * (1.0 - _CLOCK_DROP_FRACTION):
                return sample
            maximum_clock = max(maximum_clock, clock)
        return None

    @classmethod
    def _clock_verdict(
        cls,
        gpu_index: int,
        samples: tuple[GpuSample, ...],
        reported: tuple[GpuSample, ...],
    ) -> ThrottleVerdict:
        """Assess clock evidence after driver-reported active reasons have been ruled out."""
        clock_samples = tuple(sample for sample in samples if is_supported(sample.core_clock_mhz))
        drop = cls._substantial_clock_drop(samples)
        count = len(clock_samples)
        if count < _MIN_CLOCK_SAMPLES:
            state, reason, confidence = ThrottleState.UNKNOWN, "insufficient_clock_data", 0.0
        elif drop is None and reported and len(reported) == len(samples):
            state = ThrottleState.NOT_SUSPECTED
            reason, confidence = "driver_reported_no_throttling", 0.95
            count = len(reported)
        elif drop is None:
            state, reason, confidence = (
                ThrottleState.NOT_SUSPECTED,
                "no_substantial_clock_drop",
                0.6,
            )
        elif is_supported(drop.temperature_c) and drop.temperature_c >= _HIGH_TEMPERATURE_C:
            state, reason, confidence = (
                ThrottleState.SUSPECTED,
                "clock_drop_with_high_temperature",
                0.8,
            )
        elif (
            is_supported(drop.power_watts)
            and is_supported(drop.power_limit_watts)
            and drop.power_limit_watts > 0
            and drop.power_watts / drop.power_limit_watts >= _POWER_LIMIT_FRACTION
        ):
            state, reason, confidence = (
                ThrottleState.SUSPECTED,
                "clock_drop_near_power_limit",
                0.75,
            )
        elif (
            is_supported(drop.utilization_percent)
            and drop.utilization_percent <= _LOW_UTILIZATION_PERCENT
        ):
            state, reason, confidence = (
                ThrottleState.NOT_SUSPECTED,
                "clock_drop_with_low_utilization",
                0.75,
            )
        else:
            state, reason, confidence = (
                ThrottleState.UNKNOWN,
                "clock_drop_without_correlating_evidence",
                0.25,
            )
        return ThrottleVerdict(gpu_index, state, reason, confidence, count)
