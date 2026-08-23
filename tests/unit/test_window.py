"""Derived per-device statistics, energy estimates, and throttle heuristic tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from baseaicore import UNSUPPORTED, Measurement, ValidationError

from sweatmeter import GpuSample, TelemetrySnapshot
from sweatmeter.window import TelemetryWindow, ThrottleState, ThrottleVerdict, WindowMetric

_START = datetime(2026, 8, 22, 21, 0, tzinfo=UTC)


def _gpu(  # noqa: PLR0913 — one helper exposes the independent GPU sensors under test
    *,
    index: int = 0,
    vram: Measurement = UNSUPPORTED,
    power: Measurement = UNSUPPORTED,
    power_limit: Measurement = UNSUPPORTED,
    temperature: Measurement = UNSUPPORTED,
    utilization: Measurement = UNSUPPORTED,
    clock: Measurement = UNSUPPORTED,
    reasons: tuple[str, ...] = (),
    reasons_available: bool = False,
) -> GpuSample:
    return GpuSample(
        index=index,
        vram_used_bytes=vram,
        power_watts=power,
        power_limit_watts=power_limit,
        temperature_c=temperature,
        utilization_percent=utilization,
        core_clock_mhz=clock,
        throttle_reasons=reasons,
        throttle_reasons_available=reasons_available,
    )


def _snapshot(seconds: float, *gpus: GpuSample) -> TelemetrySnapshot:
    return TelemetrySnapshot(timestamp=_START + timedelta(seconds=seconds), gpus=tuple(gpus))


def test_empty_window_returns_unsupported_and_zero_evidence() -> None:
    window = TelemetryWindow([])

    assert window.sample_count() == 0
    assert window.peak_vram_bytes() is UNSUPPORTED
    assert window.mean_power_watts() is UNSUPPORTED
    assert window.energy_joules() is UNSUPPORTED
    assert window.max_temperature_c() is UNSUPPORTED
    assert window.supported_sample_count(WindowMetric.PEAK_VRAM_BYTES) == 0
    verdict = window.suspected_throttling()
    assert verdict.verdict is ThrottleState.UNKNOWN
    assert verdict.suspected is None
    assert verdict.supported_samples == 0


def test_nonempty_all_unsupported_window_never_fabricates_zero() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu()), _snapshot(1, _gpu())])

    assert window.sample_count() == 2
    assert window.peak_vram_bytes() is UNSUPPORTED
    assert window.mean_power_watts() is UNSUPPORTED
    assert window.energy_joules() is UNSUPPORTED
    assert window.max_temperature_c() is UNSUPPORTED
    for metric in (
        WindowMetric.PEAK_VRAM_BYTES,
        WindowMetric.MEAN_POWER_WATTS,
        WindowMetric.ENERGY_JOULES,
        WindowMetric.MAX_TEMPERATURE_C,
    ):
        assert window.supported_sample_count(metric) == 0


def test_statistics_exclude_unsupported_values_and_report_counts() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(vram=1_000, power=100.0, temperature=60.0)),
            _snapshot(1, _gpu(vram=UNSUPPORTED, power=UNSUPPORTED, temperature=65.0)),
            _snapshot(2, _gpu(vram=4_000, power=200.0, temperature=UNSUPPORTED)),
        ]
    )

    assert window.peak_vram_bytes() == 4_000
    assert window.mean_power_watts() == 150.0
    assert window.max_temperature_c() == 65.0
    assert window.supported_sample_count("peak_vram_bytes") == 2
    assert window.supported_sample_count(WindowMetric.MEAN_POWER_WATTS) == 2
    assert window.supported_sample_count(WindowMetric.MAX_TEMPERATURE_C) == 2


def test_energy_estimate_uses_real_irregular_timestamps_and_ignores_nominal_interval() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(power=100.0)),
            _snapshot(2, _gpu(power=200.0)),
            _snapshot(5, _gpu(power=50.0)),
            _snapshot(9, _gpu(power=999.0)),
        ]
    )

    # Left-rectangle estimate: 100*2 + 200*3 + 50*4. The final 999 W has no duration.
    assert window.energy_joules() == 1_000.0
    assert window.supported_sample_count(WindowMetric.ENERGY_JOULES) == 3
    assert "estimate" in (TelemetryWindow.energy_joules.__doc__ or "").casefold()


def test_energy_skips_unsupported_intervals_instead_of_treating_them_as_zero() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(power=UNSUPPORTED)),
            _snapshot(2, _gpu(power=200.0)),
            _snapshot(5, _gpu(power=UNSUPPORTED)),
            _snapshot(9, _gpu(power=50.0)),
        ]
    )

    assert window.energy_joules() == 600.0
    assert window.supported_sample_count(WindowMetric.ENERGY_JOULES) == 1


def test_real_zero_power_interval_returns_zero_not_unsupported() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(power=0.0)), _snapshot(2, _gpu(power=100.0))])

    assert window.energy_joules() == 0.0
    assert window.supported_sample_count(WindowMetric.ENERGY_JOULES) == 1


def test_window_sorts_out_of_order_samples_before_energy_integration() -> None:
    window = TelemetryWindow(
        [
            _snapshot(5, _gpu(power=999.0)),
            _snapshot(0, _gpu(power=100.0)),
            _snapshot(2, _gpu(power=50.0)),
        ]
    )

    assert window.energy_joules() == 350.0
    assert window.sample_count() == 3


def test_equal_timestamps_do_not_create_negative_or_fabricated_energy() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(power=100.0)), _snapshot(0, _gpu(power=200.0))])

    assert window.energy_joules() is UNSUPPORTED
    assert window.supported_sample_count(WindowMetric.ENERGY_JOULES) == 0


def test_multi_gpu_derived_metrics_remain_per_device() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(index=0, vram=1_000, power=100), _gpu(index=1, vram=8_000, power=20)),
            _snapshot(2, _gpu(index=0, vram=2_000, power=200), _gpu(index=1, vram=9_000, power=40)),
        ]
    )

    assert window.peak_vram_bytes(0) == 2_000
    assert window.peak_vram_bytes(1) == 9_000
    assert window.mean_power_watts(0) == 150.0
    assert window.mean_power_watts(1) == 30.0
    assert window.energy_joules(0) == 200.0
    assert window.energy_joules(1) == 40.0


def test_absent_device_is_unsupported_never_zero() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(index=1, power=100))])

    assert window.peak_vram_bytes(0) is UNSUPPORTED
    assert window.mean_power_watts(0) is UNSUPPORTED
    assert window.energy_joules(0) is UNSUPPORTED
    assert window.max_temperature_c(0) is UNSUPPORTED


def test_driver_reported_throttle_reason_is_high_confidence() -> None:
    window = TelemetryWindow(
        [
            _snapshot(
                0,
                _gpu(
                    clock=2_000,
                    reasons=("hw_thermal_slowdown",),
                    reasons_available=True,
                ),
            )
        ]
    )

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.SUSPECTED
    assert verdict.suspected is True
    assert verdict.reason == "driver_reported:hw_thermal_slowdown"
    assert verdict.confidence == 1.0
    assert verdict.supported_samples == 1


def test_clock_drop_with_high_temperature_is_suspected() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(clock=2_000, temperature=65, utilization=95)),
            _snapshot(1, _gpu(clock=1_600, temperature=85, utilization=95)),
        ]
    )

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.SUSPECTED
    assert verdict.reason == "clock_drop_with_high_temperature"
    assert verdict.supported_samples == 2


def test_clock_drop_near_power_limit_is_suspected() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(clock=2_000, power=100, power_limit=200, utilization=90)),
            _snapshot(1, _gpu(clock=1_600, power=195, power_limit=200, utilization=90)),
        ]
    )

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.SUSPECTED
    assert verdict.reason == "clock_drop_near_power_limit"


def test_clock_drop_at_low_utilization_is_not_suspected() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(clock=2_000, utilization=95)),
            _snapshot(1, _gpu(clock=1_600, utilization=20)),
        ]
    )

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.NOT_SUSPECTED
    assert verdict.suspected is False
    assert verdict.reason == "clock_drop_with_low_utilization"


def test_missing_clock_data_yields_unknown_not_false() -> None:
    verdict = TelemetryWindow([_snapshot(0, _gpu()), _snapshot(1, _gpu())]).suspected_throttling()

    assert verdict.verdict is ThrottleState.UNKNOWN
    assert verdict.suspected is None
    assert verdict.reason == "insufficient_clock_data"
    assert verdict.confidence == 0.0


def test_stable_clocks_with_driver_visibility_are_not_suspected() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(clock=2_000, reasons_available=True)),
            _snapshot(1, _gpu(clock=1_900, reasons_available=True)),
        ]
    )

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.NOT_SUSPECTED
    assert verdict.reason == "driver_reported_no_throttling"
    assert verdict.confidence == 0.95


def test_stable_clocks_without_driver_visibility_are_not_suspected_with_lower_confidence() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(clock=2_000)), _snapshot(1, _gpu(clock=1_900))])

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.NOT_SUSPECTED
    assert verdict.reason == "no_substantial_clock_drop"
    assert verdict.confidence == 0.6


def test_uncorrelated_clock_drop_remains_unknown() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(clock=2_000)), _snapshot(1, _gpu(clock=1_600))])

    verdict = window.suspected_throttling()

    assert verdict.verdict is ThrottleState.UNKNOWN
    assert verdict.reason == "clock_drop_without_correlating_evidence"


def test_gpu_idle_reason_alone_is_not_treated_as_performance_throttling() -> None:
    window = TelemetryWindow(
        [
            _snapshot(0, _gpu(clock=2_000, reasons=("gpu_idle",), reasons_available=True)),
            _snapshot(1, _gpu(clock=1_900, reasons=("gpu_idle",), reasons_available=True)),
        ]
    )

    assert window.suspected_throttling().verdict is ThrottleState.NOT_SUSPECTED


def test_throttle_supported_count_matches_verdict_evidence() -> None:
    window = TelemetryWindow([_snapshot(0, _gpu(clock=2_000)), _snapshot(1, _gpu(clock=1_900))])

    assert window.supported_sample_count(WindowMetric.SUSPECTED_THROTTLING) == 2


@pytest.mark.parametrize("gpu_index", [-1, True])
def test_every_metric_rejects_invalid_gpu_index(gpu_index: int) -> None:
    window = TelemetryWindow([])

    with pytest.raises(ValidationError, match="gpu_index"):
        window.peak_vram_bytes(gpu_index)
    with pytest.raises(ValidationError, match="gpu_index"):
        window.mean_power_watts(gpu_index)
    with pytest.raises(ValidationError, match="gpu_index"):
        window.energy_joules(gpu_index)
    with pytest.raises(ValidationError, match="gpu_index"):
        window.max_temperature_c(gpu_index)
    with pytest.raises(ValidationError, match="gpu_index"):
        window.suspected_throttling(gpu_index)


def test_supported_sample_count_rejects_unknown_metric() -> None:
    with pytest.raises(ValidationError, match="Unknown window metric"):
        TelemetryWindow([]).supported_sample_count("machine_power")


def test_window_rejects_non_snapshot_element() -> None:
    with pytest.raises(ValidationError, match=r"samples\[0\]"):
        TelemetryWindow(cast(list[TelemetrySnapshot], [object()]))


@pytest.mark.parametrize(
    "verdict",
    [
        lambda: ThrottleVerdict(-1, ThrottleState.UNKNOWN, "reason", 0.0, 0),
        lambda: ThrottleVerdict(0, cast(Any, "unknown"), "reason", 0.0, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "", 0.0, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "   ", 0.0, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, cast(Any, 1), 0.0, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "reason", 1.1, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "reason", 10**1_000, 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "reason", cast(Any, "high"), 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "reason", cast(Any, True), 0),
        lambda: ThrottleVerdict(0, ThrottleState.UNKNOWN, "reason", 0.0, -1),
    ],
)
def test_throttle_verdict_validates_invariants(
    verdict: Callable[[], ThrottleVerdict],
) -> None:
    with pytest.raises(ValidationError):
        verdict()
