"""Low-overhead background sampling with explicit lifecycle and staleness."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from types import TracebackType

from baseaicore import ValidationError

from sweatmeter.collector import TelemetryCollector
from sweatmeter.safe import _safe
from sweatmeter.types import TelemetrySnapshot

type MonotonicClock = Callable[[], float]

__all__ = ["TelemetrySampler"]


def _positive_seconds(value: object, *, field: str) -> float:
    """Validate and normalize a positive finite duration."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValidationError(
            f"TelemetrySampler.{field} must be a positive finite number; got {value!r}.",
            details={"field": field, "value": value},
        )
    return float(value)


class TelemetrySampler:
    """Collect telemetry on one restartable daemon thread at a monotonic interval.

    The sampler publishes the newest snapshot atomically, optionally retains a bounded history,
    and exposes both callback and blocking-iterator consumption. A slow iterator can miss values
    that have fallen out of the configured ring buffer; memory never grows to preserve an
    unbounded backlog. Callback exceptions are isolated and logged at DEBUG by SweatMeter's shared
    boundary helper. Thread start/stop and reads are synchronized, so the instance is safe for
    concurrent consumers.

    Sampling starts immediately on each :meth:`start`, then follows monotonic deadlines. Missed
    deadlines are skipped instead of firing a burst of catch-up samples that could distort the
    workload being measured. The thread is a daemon as a final process-exit safeguard, but normal
    users should stop it explicitly or use the context manager.

    Args:
        collector: Non-raising telemetry collector called by the worker thread.
        interval_seconds: Positive finite delay between sampling deadlines.
        on_sample: Optional callback invoked after a snapshot becomes visible to readers.
        buffer_size: Optional positive maximum retained snapshot count. When omitted, only the
            newest snapshot is held.
        monotonic_clock: Injectable duration clock used for scheduling and sample age.

    Raises:
        ValidationError: If an interval, buffer size, callback, or clock is invalid.
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        *,
        interval_seconds: float = 1.0,
        on_sample: Callable[[TelemetrySnapshot], None] | None = None,
        buffer_size: int | None = None,
        monotonic_clock: MonotonicClock = time.monotonic,
    ) -> None:
        """Configure sampling without starting a thread."""
        if on_sample is not None and not callable(on_sample):
            raise ValidationError(
                f"TelemetrySampler.on_sample must be callable or None; got {on_sample!r}.",
                details={"field": "on_sample"},
            )
        if not callable(monotonic_clock):
            raise ValidationError(
                "TelemetrySampler.monotonic_clock must be callable.",
                details={"field": "monotonic_clock"},
            )
        if buffer_size is not None and (
            isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0
        ):
            raise ValidationError(
                f"TelemetrySampler.buffer_size must be a positive integer or None; got "
                f"{buffer_size!r}.",
                details={"field": "buffer_size", "value": buffer_size},
            )

        self._collector = collector
        self._interval_seconds = _positive_seconds(interval_seconds, field="interval_seconds")
        self._on_sample = on_sample
        self._monotonic_clock = monotonic_clock
        self._history: deque[tuple[int, TelemetrySnapshot, float]] = deque(
            maxlen=1 if buffer_size is None else buffer_size
        )
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0

    def start(self) -> None:
        """Start sampling, or do nothing when the worker is already running.

        A stopped or failed sampler can be restarted. Retained history and sequence numbers remain
        intact across restarts so existing snapshots are never relabelled.
        """
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="sweatmeter-sampler",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Request shutdown and wait for the worker up to a bounded duration.

        Calling ``stop`` before ``start`` or after shutdown is safe. A callback may also stop its
        own sampler; that path sets the stop request without trying to join the current thread.

        Args:
            timeout: Positive finite maximum seconds to wait for another worker thread.

        Raises:
            ValidationError: If ``timeout`` is not positive and finite.
            TimeoutError: If a collector call remains blocked beyond the timeout.
        """
        timeout_seconds = _positive_seconds(timeout, field="timeout")
        with self._condition:
            thread = self._thread
            self._stop_event.set()
            self._condition.notify_all()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(
                f"TelemetrySampler worker did not stop within {timeout_seconds:g} seconds; "
                "a telemetry boundary may be blocked."
            )

    def is_running(self) -> bool:
        """Return whether the worker thread is currently alive."""
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def latest(self) -> TelemetrySnapshot | None:
        """Return the newest snapshot, or ``None`` before the first successful collection."""
        with self._condition:
            return self._history[-1][1] if self._history else None

    def latest_age_seconds(self) -> float | None:
        """Return the newest snapshot's monotonic age, or ``None`` when no sample exists.

        The age is based on local receipt time rather than wall-clock timestamp subtraction, so an
        NTP correction cannot make a stale sample look current or produce a negative duration.
        """
        with self._condition:
            if not self._history:
                return None
            sampled_at = self._history[-1][2]
        return max(0.0, self._monotonic_now() - sampled_at)

    def buffered(self) -> tuple[TelemetrySnapshot, ...]:
        """Return retained snapshots from oldest to newest as an immutable tuple."""
        with self._condition:
            return tuple(snapshot for _sequence, snapshot, _sampled_at in self._history)

    def __iter__(self) -> Iterator[TelemetrySnapshot]:
        """Yield retained and future snapshots until the sampler stops.

        Each iterator owns its sequence cursor. If its consumer is slower than a bounded ring
        buffer, overwritten snapshots are skipped honestly rather than copied into an unbounded
        per-consumer queue.
        """
        next_sequence = 1
        while True:
            with self._condition:
                available = tuple(item for item in self._history if item[0] >= next_sequence)
                while not available and self._thread_is_alive_locked():
                    self._condition.wait()
                    available = tuple(item for item in self._history if item[0] >= next_sequence)
                if not available:
                    return
            for sequence, snapshot, _sampled_at in available:
                next_sequence = sequence + 1
                yield snapshot

    def __enter__(self) -> TelemetrySampler:
        """Start sampling and return this sampler."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop sampling on every context-manager exit path."""
        del exc_type, exc_value, traceback
        self.stop()

    def _thread_is_alive_locked(self) -> bool:
        """Return worker liveness while the caller holds ``_condition``."""
        return self._thread is not None and self._thread.is_alive()

    def _monotonic_now(self) -> float:
        """Read a valid monotonic value, falling back to the standard clock if injected badly."""
        value = _safe(self._monotonic_clock, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return time.monotonic()
        return float(value)

    def _publish(self, snapshot: TelemetrySnapshot, *, sampled_at: float) -> None:
        """Publish atomically, wake iterators, and isolate the optional callback."""
        with self._condition:
            self._sequence += 1
            self._history.append((self._sequence, snapshot, sampled_at))
            self._condition.notify_all()
        callback = self._on_sample
        if callback is not None:
            _safe(lambda: callback(snapshot), None)

    def _run(self) -> None:
        """Collect on monotonic deadlines and always wake iterators when exiting."""
        deadline = self._monotonic_now()
        try:
            while not self._stop_event.is_set():
                snapshot = self._collector.snapshot()
                sampled_at = self._monotonic_now()
                self._publish(snapshot, sampled_at=sampled_at)

                deadline += self._interval_seconds
                now = self._monotonic_now()
                if deadline <= now:
                    missed_intervals = math.floor((now - deadline) / self._interval_seconds) + 1
                    deadline += missed_intervals * self._interval_seconds
                if self._stop_event.wait(max(0.0, deadline - now)):
                    return
        finally:
            with self._condition:
                self._condition.notify_all()
