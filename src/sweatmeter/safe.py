"""Narrow failure isolation for operating-system telemetry boundaries."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import overload

from baseaicore import UNSUPPORTED, Unsupported

_LOGGER = logging.getLogger(__name__)

__all__ = ["_safe", "_safe_call"]


def _safe_call[T, D](fn: Callable[[], T], default: D) -> tuple[T | D, Exception | None]:
    """Call ``fn``, returning its result or ``default`` together with the failure.

    This holds SweatMeter's **only** broad catch. It belongs at an operating-system or sensor
    boundary, where one unreadable source must degrade one field instead of aborting a snapshot.
    Returning the exception rather than discarding it lets a caller classify the failure into a
    diagnostic reason without opening a second ``except Exception`` anywhere else in the package
    (Coding Standards §6).

    ``KeyboardInterrupt`` and ``SystemExit`` inherit directly from ``BaseException``, not
    ``Exception``, and deliberately continue to propagate.

    Args:
        fn: Zero-argument boundary operation.
        default: Honest degraded value used when the call fails.

    Returns:
        ``(result, None)`` on success, or ``(default, exception)`` after an ordinary exception.
    """
    try:
        return (fn(), None)
    except Exception as exc:
        operation = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        _LOGGER.debug("telemetry.source_failed", extra={"operation": operation}, exc_info=True)
        return (default, exc)


@overload
def _safe[T](fn: Callable[[], T]) -> T | Unsupported: ...


@overload
def _safe[T, D](fn: Callable[[], T], default: D) -> T | D: ...


def _safe[T, D](fn: Callable[[], T], default: D | Unsupported = UNSUPPORTED) -> T | D | Unsupported:
    """Call ``fn`` and convert an ordinary boundary failure to ``default``.

    The sanctioned degrade-one-field helper, used wherever the caller needs the value but not the
    reason. It delegates to :func:`_safe_call`, so the package still contains exactly one broad
    catch. The exception is retained in a DEBUG record for diagnosis.

    Args:
        fn: Zero-argument boundary operation.
        default: Honest degraded value. Defaults to ``UNSUPPORTED``.

    Returns:
        The callable's result, or ``default`` after an ordinary exception.
    """
    value, _failure = _safe_call(fn, default)
    return value
