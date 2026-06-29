"""Opt-in timing probes for MockSoc SimEngine diagnostics."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

PROFILE_ENV = "ZCU_AUTOFLUXDEP_PROFILE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def profiling_enabled(logger: logging.Logger) -> bool:
    value = os.environ.get(PROFILE_ENV, "").strip().lower()
    return value in _TRUE_VALUES or logger.isEnabledFor(logging.DEBUG)


def perf_now() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def current_thread_label() -> str:
    thread = threading.current_thread()
    return f"{thread.name}/{thread.ident}"


@dataclass
class PerfStats:
    label: str
    logger: logging.Logger
    interval_s: float = 1.0
    slow_ms: float | None = None
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    window_start: float = field(default_factory=time.perf_counter)

    def record(self, duration_ms: float, *, detail: str = "") -> None:
        if not profiling_enabled(self.logger):
            return

        self.count += 1
        self.total_ms += duration_ms
        self.max_ms = max(self.max_ms, duration_ms)

        if self.slow_ms is not None and duration_ms >= self.slow_ms:
            self.logger.warning(
                "autofluxdep profile slow %s: duration_ms=%.1f thread=%s%s",
                self.label,
                duration_ms,
                current_thread_label(),
                f" {detail}" if detail else "",
            )

        now = time.perf_counter()
        if now - self.window_start < self.interval_s:
            return

        avg_ms = self.total_ms / max(1, self.count)
        self.logger.info(
            "autofluxdep profile %s: count=%d avg_ms=%.1f max_ms=%.1f thread=%s",
            self.label,
            self.count,
            avg_ms,
            self.max_ms,
            current_thread_label(),
        )
        self.reset(now)

    def reset(self, now: float | None = None) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.window_start = time.perf_counter() if now is None else now
