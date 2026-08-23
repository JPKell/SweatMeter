"""Timing, callback, iterator, buffering, and lifecycle tests for Phase 3 sampling."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import cast

import pytest
from baseaicore import ValidationError

from sweatmeter import NullGpuReader, NullHostReader, TelemetryCollector, TelemetrySampler
from sweatmeter.types import TelemetrySnapshot

_NOW = datetime(2026, 8, 22, 20, 0, tzinfo=UTC)


class SequenceCollector(TelemetryCollector):
    def __init__(self, cpu_values: Sequence[float] = (1.0, 2.0, 3.0)) -> None:
        self._cpu_values = tuple(cpu_values)
        self._position = 0
        self._sequence_lock = threading.Lock()

    def snapshot(self) -> TelemetrySnapshot:
        with self._sequence_lock:
            position = self._position
            self._position += 1
        return TelemetrySnapshot(
            timestamp=_NOW + timedelta(milliseconds=position),
            cpu_percent=self._cpu_values[position % len(self._cpu_values)],
        )


class BlockingCollector(TelemetryCollector):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def snapshot(self) -> TelemetrySnapshot:
        self.entered.set()
        self.release.wait(2.0)
        return TelemetrySnapshot(timestamp=_NOW)


class FakeMonotonicClock:
    def __init__(self, value: float) -> None:
        self._value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


def test_sampler_starts_immediately_publishes_latest_and_stops() -> None:
    received: list[TelemetrySnapshot] = []
    enough_samples = threading.Event()

    def receive(snapshot: TelemetrySnapshot) -> None:
        received.append(snapshot)
        if len(received) >= 3:
            enough_samples.set()

    sampler = TelemetrySampler(
        SequenceCollector(), interval_seconds=0.005, on_sample=receive, buffer_size=4
    )

    sampler.start()
    assert enough_samples.wait(1.0)
    sampler.stop(timeout=1.0)

    assert sampler.is_running() is False
    assert sampler.latest() == received[-1]
    assert sampler.buffered()[-1] == received[-1]


def test_sampler_follows_interval_without_bursting() -> None:
    starts: list[float] = []
    enough_samples = threading.Event()

    def record(_snapshot: TelemetrySnapshot) -> None:
        starts.append(time.monotonic())
        if len(starts) == 4:
            enough_samples.set()

    sampler = TelemetrySampler(SequenceCollector(), interval_seconds=0.02, on_sample=record)

    sampler.start()
    assert enough_samples.wait(1.0)
    sampler.stop(timeout=1.0)
    intervals = [later - earlier for earlier, later in pairwise(starts)]

    assert len(intervals) >= 3
    assert all(0.012 <= interval <= 0.06 for interval in intervals[:3])


def test_sampler_skips_missed_deadlines_instead_of_bursting_catch_up_samples() -> None:
    interval = 0.02
    clock = FakeMonotonicClock(1_000.0)
    starts: list[float] = []
    enough_samples = threading.Event()

    def lag_behind(_snapshot: TelemetrySnapshot) -> None:
        starts.append(time.monotonic())
        # Simulate a collection that overran ten deadlines while the worker was busy.
        clock.advance(interval * 10)
        if len(starts) == 4:
            enough_samples.set()

    sampler = TelemetrySampler(
        SequenceCollector(),
        interval_seconds=interval,
        on_sample=lag_behind,
        monotonic_clock=clock,
    )

    sampler.start()
    assert enough_samples.wait(2.0)
    sampler.stop(timeout=1.0)
    intervals = [later - earlier for earlier, later in pairwise(starts)]

    assert len(intervals) >= 3
    assert all(gap >= interval / 2 for gap in intervals), intervals


def test_sampler_falls_back_when_the_injected_monotonic_clock_is_invalid() -> None:
    def broken_clock() -> float:
        return cast(float, "not-a-duration")

    sampled = threading.Event()
    sampler = TelemetrySampler(
        SequenceCollector(),
        interval_seconds=0.005,
        on_sample=lambda _snapshot: sampled.set(),
        monotonic_clock=broken_clock,
    )

    sampler.start()
    assert sampled.wait(1.0)
    sampler.stop(timeout=1.0)
    age = sampler.latest_age_seconds()

    assert sampler.latest() is not None
    assert age is not None
    assert age >= 0.0


def test_raising_callback_does_not_kill_worker() -> None:
    calls = 0
    enough_samples = threading.Event()

    def broken_callback(_snapshot: TelemetrySnapshot) -> None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            enough_samples.set()
        raise RuntimeError("consumer failed")

    sampler = TelemetrySampler(
        SequenceCollector(), interval_seconds=0.003, on_sample=broken_callback
    )

    sampler.start()
    assert enough_samples.wait(1.0)
    assert sampler.is_running() is True
    sampler.stop(timeout=1.0)

    assert calls >= 3


def test_callback_can_stop_its_own_sampler_without_self_join() -> None:
    stopped_from_callback = threading.Event()
    sampler: TelemetrySampler

    def stop_from_callback(_snapshot: TelemetrySnapshot) -> None:
        sampler.stop(timeout=1.0)
        stopped_from_callback.set()

    sampler = TelemetrySampler(
        SequenceCollector(), interval_seconds=0.01, on_sample=stop_from_callback
    )

    sampler.start()
    assert stopped_from_callback.wait(1.0)
    deadline = time.monotonic() + 1.0
    while sampler.is_running() and time.monotonic() < deadline:
        threading.Event().wait(0.001)

    assert sampler.is_running() is False


def test_bounded_buffer_discards_oldest_without_growing() -> None:
    enough_samples = threading.Event()
    received = 0

    def count(_snapshot: TelemetrySnapshot) -> None:
        nonlocal received
        received += 1
        if received >= 6:
            enough_samples.set()

    sampler = TelemetrySampler(
        SequenceCollector(), interval_seconds=0.002, on_sample=count, buffer_size=3
    )

    sampler.start()
    assert enough_samples.wait(1.0)
    sampler.stop(timeout=1.0)

    assert len(sampler.buffered()) == 3
    assert sampler.latest() == sampler.buffered()[-1]
    assert list(sampler) == list(sampler.buffered())


def test_iterator_yields_future_sample_while_running() -> None:
    sampler = TelemetrySampler(SequenceCollector(), interval_seconds=0.05)

    sampler.start()
    first = next(iter(sampler))
    sampler.stop(timeout=1.0)

    assert first.cpu_percent == 1.0


def test_latest_exposes_monotonic_age_with_fake_clock() -> None:
    clock = FakeMonotonicClock(10.0)
    sampled = threading.Event()
    sampler = TelemetrySampler(
        SequenceCollector(),
        interval_seconds=10.0,
        on_sample=lambda _snapshot: sampled.set(),
        monotonic_clock=clock,
    )

    sampler.start()
    assert sampled.wait(1.0)
    sampler.stop(timeout=1.0)
    clock.advance(2.5)

    assert sampler.latest_age_seconds() == 2.5


def test_context_manager_stops_worker_when_body_raises() -> None:
    sampler = TelemetrySampler(SequenceCollector(), interval_seconds=0.01)

    with pytest.raises(RuntimeError, match="body failed"), sampler:
        raise RuntimeError("body failed")

    assert sampler.is_running() is False


def test_sampler_is_restartable_and_start_is_idempotent() -> None:
    sampled = threading.Event()
    sampler = TelemetrySampler(
        SequenceCollector(), interval_seconds=0.02, on_sample=lambda _snapshot: sampled.set()
    )

    sampler.start()
    sampler.start()
    assert sampler.is_running() is True
    assert sampled.wait(1.0)
    sampler.stop(timeout=1.0)
    first_latest = sampler.latest()

    sampled.clear()
    sampler.start()
    assert sampled.wait(1.0)
    sampler.stop(timeout=1.0)

    assert sampler.latest() != first_latest


def test_one_hundred_start_stop_cycles_leave_no_worker_thread() -> None:
    baseline = sum(thread.name == "sweatmeter-sampler" for thread in threading.enumerate())

    for _cycle in range(100):
        sampler = TelemetrySampler(SequenceCollector(), interval_seconds=10.0)
        sampler.start()
        sampler.stop(timeout=1.0)

    remaining = sum(thread.name == "sweatmeter-sampler" for thread in threading.enumerate())
    assert remaining == baseline


def test_stop_reports_blocked_collector_and_can_finish_after_release() -> None:
    collector = BlockingCollector()
    sampler = TelemetrySampler(collector, interval_seconds=10.0)
    sampler.start()
    assert collector.entered.wait(1.0)

    with pytest.raises(TimeoutError, match="did not stop"):
        sampler.stop(timeout=0.01)

    collector.release.set()
    sampler.stop(timeout=1.0)
    assert sampler.is_running() is False


def test_stop_before_start_and_empty_accessors_are_safe() -> None:
    sampler = TelemetrySampler(SequenceCollector())

    sampler.stop()

    assert sampler.latest() is None
    assert sampler.latest_age_seconds() is None
    assert sampler.buffered() == ()
    assert list(sampler) == []


@pytest.mark.parametrize(
    "build",
    [
        lambda collector: TelemetrySampler(collector, interval_seconds=0),
        lambda collector: TelemetrySampler(collector, interval_seconds=float("nan")),
        lambda collector: TelemetrySampler(collector, buffer_size=0),
        lambda collector: TelemetrySampler(collector, buffer_size=True),
        lambda collector: TelemetrySampler(collector, on_sample=42),  # type: ignore[arg-type]
        lambda collector: TelemetrySampler(collector, monotonic_clock=42),  # type: ignore[arg-type]
    ],
)
def test_sampler_rejects_invalid_configuration(
    build: Callable[[TelemetryCollector], TelemetrySampler],
) -> None:
    with pytest.raises(ValidationError):
        build(SequenceCollector())


def test_stop_rejects_invalid_timeout_without_starting() -> None:
    sampler = TelemetrySampler(
        TelemetryCollector(host=NullHostReader(), gpu=NullGpuReader()), interval_seconds=1.0
    )

    with pytest.raises(ValidationError, match="timeout"):
        sampler.stop(timeout=0)
