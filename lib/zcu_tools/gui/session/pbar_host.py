"""Progress-bar host: the worker-side bar and the main-thread bar model.

Both classes here are **Qt-free**. The cross-thread marshal lives in a
``ProgressTransport`` (a session port; the Qt implementation is a driven
adapter), and the container/ownership logic lives in a progress service. This
file holds only the two ends of a single bar:

  - ``ProgressBar`` (worker side): a ``BaseProgressBar`` the experiment code
    drives via the ``make_pbar`` factory. It is **pure transport + throttle** —
    it forwards raw (operation_id, handle_id, label, total, n) as
    ``ProgressEvent``s and computes nothing (no format, no timing). It never sees
    a container; the consumer routes by the ids it carries.
  - ``ProgressBarModel`` (main thread, the SSOT): one mutable object per live
    bar holding raw state (label / total / n / start_time). Derived values
    (format string, percent, elapsed/remaining) are **methods computed live on
    query** — every reader (widget render, agent ``run.progress``) asks this
    object, so there is a single source of truth and no stale formatted string.

App-agnostic session-core infra: every measurement-session app (measure /
autofluxdep) reuses these for per-operation progress.
"""

from __future__ import annotations

import time

from zcu_tools.gui.session.ports import (
    ProgressEvent,
    ProgressEventKind,
    ProgressTransport,
)
from zcu_tools.progress_bar.base import BaseProgressBar, ProgressTotal, ProgressValue

# Float totals are rendered on an integer Qt bar by scaling to this resolution.
_FLOAT_SCALE = 10000
_PUBLISH_INTERVAL = 0.033  # ~30 fps cap for cross-thread progress updates


def _fmt_seconds(secs: float) -> str:
    minutes, seconds = divmod(int(max(0.0, secs)), 60)
    return f"{minutes}:{seconds:02d}"


class ProgressBarModel:
    """Main-thread SSOT for one live progress bar.

    Holds raw state; format / percent / timing are computed live on each query
    (so elapsed/remaining reflect wall-clock at read time, not a frozen string).
    ``start_time`` is stamped by the main thread when the bar is first created.
    """

    def __init__(self, label: str, total: ProgressTotal, start_time: float) -> None:
        self.label = label
        self.total = total
        self.n: ProgressValue = 0
        self.start_time = start_time

    def set_n(self, n: ProgressValue) -> None:
        self.n = n

    def set_total(self, total: ProgressTotal) -> None:
        self.total = total

    def set_label(self, label: str) -> None:
        self.label = label

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.start_time)

    def remaining(self) -> float | None:
        total, n, elapsed = self.total, self.n, self.elapsed()
        if total is None or total <= 0 or n <= 0 or elapsed <= 0:
            return None
        rate = float(n) / elapsed
        return (float(total) - float(n)) / rate if rate > 0 else None

    def qt_maximum(self) -> int:
        if self.total is None or self.total == 0:
            return 0
        if isinstance(self.total, int):
            return self.total
        return _FLOAT_SCALE

    def qt_value(self) -> int:
        total = self.total
        if isinstance(total, int):
            return int(round(self.n))
        if isinstance(total, float) and total > 0:
            return int(round(float(self.n) / total * _FLOAT_SCALE))
        return 0

    def percent(self) -> float | None:
        maximum = self.qt_maximum()
        if maximum == 0:
            return None
        return round(self.qt_value() / maximum * 100, 1)

    def format(self) -> str:
        """Human-readable bar string, e.g. ``Rounds 23/100 [0:25<1:15]``."""
        elapsed = self.elapsed()
        remaining = self.remaining()
        if remaining is not None:
            time_part = f"[{_fmt_seconds(elapsed)}<{_fmt_seconds(remaining)}]"
        else:
            time_part = f"[{_fmt_seconds(elapsed)}]"
        prefix = f"{self.label} " if self.label else ""
        if isinstance(self.total, int):
            return f"{prefix}%v/%m {time_part}"
        return f"{prefix}{time_part}"


class ProgressBar(BaseProgressBar):
    """Worker-side bar: pure raw-forward + throttle, no format/timing here.

    Bound to a fixed ``(operation_id, handle_id)`` at construction; every event
    it emits carries those ids, so the consumer always routes it to the same
    container/bar.
    """

    def __init__(
        self,
        transport: ProgressTransport,
        operation_id: int,
        handle_id: int,
        label: str = "",
        total: ProgressTotal = None,
        leave: bool = True,
        disabled: bool = False,
    ) -> None:
        self._transport = transport
        self._operation_id = operation_id
        self._handle_id = handle_id
        self._label = label
        self._total = total
        self._leave = leave
        self._disabled = disabled
        self._n: ProgressValue = 0
        self._last_publish: float = 0.0
        if not disabled:
            self._emit(ProgressEventKind.CREATE)

    def _emit(self, kind: ProgressEventKind) -> None:
        self._transport.emit(
            ProgressEvent(
                operation_id=self._operation_id,
                handle_id=self._handle_id,
                kind=kind,
                label=self._label,
                total=self._total,
                n=self._n,
            )
        )

    def _publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < _PUBLISH_INTERVAL:
            return
        self._last_publish = now
        self._emit(ProgressEventKind.UPDATE)

    def set_description(self, description: str) -> None:
        self._label = description
        if not self._disabled:
            self._publish(force=True)

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value
        if not self._disabled:
            self._publish()

    def set_progress(self, value: ProgressValue) -> None:
        self._n = value
        if not self._disabled:
            self._publish()

    def reset(self) -> None:
        self._n = 0
        if not self._disabled:
            # Re-create so the main thread re-stamps start_time.
            self._emit(ProgressEventKind.CREATE)

    def refresh(self) -> None:
        if not self._disabled:
            self._publish(force=True)

    def close(self) -> None:
        if not self._disabled and not self._leave:
            self._emit(ProgressEventKind.CLOSE)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        if not self._disabled:
            # A total change is an in-place mutation of an existing bar, not a
            # re-create: emit UPDATE (which carries the new total) so the main
            # thread keeps the same model + widget (no flicker), preserving the
            # bar's start_time. force=True so the new total is not throttled away.
            self._publish(force=True)

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._label
