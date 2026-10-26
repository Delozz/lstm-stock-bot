"""Retry wrapper for Alpaca and Finnhub calls (FR-12)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
RETRY_DELAYS = (1, 2, 4, 8, 16)


def status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from alpaca-py, finnhub, or requests exceptions."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def is_retryable(exc: BaseException) -> bool:
    status = status_of(exc)
    return status is not None and (status == 429 or 500 <= status < 600)


def with_retries[T](
    fn: Callable[[], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
    delays: tuple[float, ...] = RETRY_DELAYS,
) -> T:
    """Call fn, retrying on 429/5xx after each delay; the final failure propagates."""
    for delay in delays:
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            sleep(delay)
    return fn()
