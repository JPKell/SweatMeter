"""Narrow failure isolation for operating-system telemetry boundaries."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import overload

from baseaicore import UNSUPPORTED, Unsupported

_LOGGER = logging.getLogger(__name__)

__all__ = ["_safe"]


@overload
def _safe[T](fn: Callable[[], T]) -> T | Unsupported: ...


@overload
def _safe[T, D](fn: Callable[[], T], default: D) -> T | D: ...


def _safe[T, D](fn: Callable[[], T], default: D | Unsupported = UNSUPPORTED) -> T | D | Unsupported:
    """Call ``fn`` and convert an ordinary boundary failure to ``default``.

    This is the one sanctioned broad catch in SweatMeter. It belongs at an operating-system or
    sensor boundary, where one unreadable source must degrade one field instead of aborting a
    snapshot. The exception is retained in a DEBUG record for diagnosis.

    ``KeyboardInterrupt`` and ``SystemExit`` inherit directly from ``BaseException``, not
    ``Exception``, and deliberately continue to propagate.

    Args:
        fn: Zero-argument boundary operation.
        default: Honest degraded value. Defaults to ``UNSUPPORTED``.

    Returns:
        The callable's result, or ``default`` after an ordinary exception.
    """
    try:
        return fn()
    except Exception:
        operation = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        _LOGGER.debug("Telemetry source failed: %s", operation, exc_info=True)
        return default
