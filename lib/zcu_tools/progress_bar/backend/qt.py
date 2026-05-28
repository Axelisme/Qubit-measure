"""Qt progress bar backend — bridges BaseProgressBar API to _ProgressStack via Qt signals.

Because RunWorker runs on a QThread, widget operations must be dispatched to
the main thread. This module uses Qt signals (emitted from any thread, received
in the main thread) to push/pop/update the _ProgressStack widget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from ..base import BaseProgressBar, ProgressTotal, ProgressValue

if TYPE_CHECKING:
    from zcu_tools.gui.ui.progress_stack import ProgressStack


@dataclass(frozen=True)
class RunProgressSnapshot:
    """Snapshot of one active progress bar, safe to read from any thread."""

    token: int
    desc: str
    n: int
    total: Optional[int]
    elapsed: float
    remaining: Optional[float]
    format: str


_FLOAT_SCALE = 10000  # maps float [0, total] → int [0, _FLOAT_SCALE]


def _is_int_total(total: object) -> bool:
    return isinstance(total, int)


class _StackBridge(QObject):
    """Lives in the main thread; receives signals and forwards to ProgressStack.

    Integer totals show exact counts (%v/%m) with a filled progress bar.
    Float totals show only timing — bar still fills proportionally via _FLOAT_SCALE.
    """

    push_requested: Signal = Signal(str, object)  # label, total (int | float | None)
    pop_requested: Signal = Signal(object)  # QProgressBar
    update_requested: Signal = Signal(object, float)  # QProgressBar, delta
    set_value_requested: Signal = Signal(object, float)  # QProgressBar, value
    set_max_requested: Signal = Signal(object, object)  # QProgressBar, total
    set_format_requested: Signal = Signal(object, str)  # QProgressBar, fmt

    def __init__(self, stack: "ProgressStack") -> None:
        super().__init__()
        self._stack = stack
        self._pending: dict[int, Any] = {}
        # per-bar: True = int mode, False = float mode
        self._int_mode: dict[int, bool] = {}
        # per-bar: float total (for computing scaled qt value)
        self._float_total: dict[int, float] = {}
        # snapshot tracking (main-thread only)
        self._bar_start_times: dict[int, float] = {}
        self._bar_snapshots: dict[int, RunProgressSnapshot] = {}
        # progress step callback (main-thread only)
        self._on_progress_step: Optional[Callable[[], None]] = None
        self._step_counter: int = 0
        self._step_interval: int = 10

        self.push_requested.connect(self._on_push)
        self.pop_requested.connect(self._on_pop)
        self.update_requested.connect(self._on_update)
        self.set_value_requested.connect(self._on_set_value)
        self.set_max_requested.connect(self._on_set_max)
        self.set_format_requested.connect(self._on_set_format)

    def _qt_max_and_mode(self, total: object) -> tuple[int, bool]:
        """Return (qt_max, is_int_mode) for a given total."""
        if total is None or total == 0:
            return 0, False  # indeterminate
        if _is_int_total(total):
            return int(total), True  # type: ignore[arg-type]
        return _FLOAT_SCALE, False

    def _on_push(self, label: str, total: object) -> None:
        qt_max, is_int = self._qt_max_and_mode(total)
        bar = self._stack.push(label, qt_max)
        key = id(bar)
        self._pending[key] = bar
        self._int_mode[key] = is_int
        if not is_int and isinstance(total, (int, float)) and total > 0:
            self._float_total[key] = float(total)
        self._bar_start_times[key] = time.monotonic()
        total_int = int(total) if isinstance(total, int) else None
        self._bar_snapshots[key] = RunProgressSnapshot(
            token=key, desc=label, n=0, total=total_int,
            elapsed=0.0, remaining=None, format=label,
        )

    def _on_pop(self, bar: object) -> None:
        key = id(bar)
        self._int_mode.pop(key, None)
        self._float_total.pop(key, None)
        self._bar_start_times.pop(key, None)
        self._bar_snapshots.pop(key, None)
        self._stack.pop(bar)  # type: ignore[arg-type]

    def _scaled_value(self, bar: object, raw: float) -> int:
        ft = self._float_total.get(id(bar))
        if ft is None or ft == 0:
            return 0
        return int(round(raw / ft * _FLOAT_SCALE))

    def _on_update(self, bar: object, delta: float) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if not isinstance(bar, QProgressBar):
            return
        key = id(bar)
        if self._int_mode.get(key, False):
            bar.setValue(bar.value() + int(round(delta)))
        elif key in self._float_total:
            bar.setValue(bar.value() + self._scaled_value(bar, delta))

    def _on_set_value(self, bar: object, value: float) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if not isinstance(bar, QProgressBar):
            return
        key = id(bar)
        if self._int_mode.get(key, False):
            bar.setValue(int(round(value)))
        elif key in self._float_total:
            bar.setValue(self._scaled_value(bar, value))

    def _on_set_max(self, bar: object, total: object) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if not isinstance(bar, QProgressBar):
            return
        key = id(bar)
        qt_max, is_int = self._qt_max_and_mode(total)
        self._int_mode[key] = is_int
        if not is_int and isinstance(total, (int, float)) and total > 0:
            self._float_total[key] = float(total)
        else:
            self._float_total.pop(key, None)
        bar.setMaximum(qt_max)

    def _on_set_format(self, bar: object, fmt: str) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if not isinstance(bar, QProgressBar):
            return
        bar.setFormat(fmt)
        key = id(bar)
        if key not in self._bar_snapshots:
            return
        start = self._bar_start_times.get(key, time.monotonic())
        elapsed = round(time.monotonic() - start, 1)
        n = bar.value() if self._int_mode.get(key, False) else 0
        total_qt = bar.maximum() if self._int_mode.get(key, False) else None
        total_int = total_qt if total_qt and total_qt > 0 else None
        remaining: Optional[float] = None
        if total_int and n > 0 and elapsed > 0:
            rate = n / elapsed
            if rate > 0:
                remaining = round((total_int - n) / rate, 1)
        prev = self._bar_snapshots[key]
        self._bar_snapshots[key] = RunProgressSnapshot(
            token=key,
            desc=prev.desc,
            n=n,
            total=total_int,
            elapsed=elapsed,
            remaining=remaining,
            format=fmt,
        )
        self._step_counter += 1
        if (
            self._on_progress_step is not None
            and self._step_counter % self._step_interval == 0
        ):
            self._on_progress_step()

    def set_progress_callback(
        self, cb: Callable[[], None], interval: int = 10
    ) -> None:
        """Register a callback invoked every ``interval`` format updates (main-thread only)."""
        self._on_progress_step = cb
        self._step_interval = max(1, interval)
        self._step_counter = 0

    def get_all_snapshots(self) -> tuple[RunProgressSnapshot, ...]:
        """Return snapshots of all currently active bars (main-thread only)."""
        return tuple(self._bar_snapshots.values())

    def pop_pending(self) -> Any:
        """Called from worker thread after push_requested; blocks until bar appears."""
        import time

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._pending:
                _, bar = self._pending.popitem()
                return bar
            time.sleep(0.005)
        raise RuntimeError("QtProgressBar: timed out waiting for push to complete")


def _fmt_seconds(secs: float) -> str:
    """Format seconds as M:SS (no hours, keeps it compact)."""
    secs = max(0.0, secs)
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


class QtProgressBar(BaseProgressBar):
    """BaseProgressBar that drives a QProgressBar on _ProgressStack via signals."""

    def __init__(
        self,
        bridge: "_StackBridge",
        label: str = "",
        total: Optional[ProgressTotal] = None,
        leave: bool = True,
        disabled: bool = False,
        **_kwargs: Any,
    ) -> None:
        import time

        self._disabled = disabled
        self._bridge = bridge
        self._label = label
        self._total: ProgressTotal = total
        self._leave = leave
        self._n: ProgressValue = 0
        self._start_time: float = time.monotonic()
        if disabled:
            self._bar: Any = None
            return
        bridge.push_requested.emit(label, total)
        self._bar = bridge.pop_pending()

    # ------------------------------------------------------------------

    def _is_int_total(self) -> bool:
        return _is_int_total(self._total)

    def _build_format(self) -> str:
        import time

        elapsed = time.monotonic() - self._start_time
        elapsed_str = _fmt_seconds(elapsed)

        n = float(self._n)
        total = float(self._total) if self._total is not None else None

        if total is not None and total > 0 and n > 0:
            rate = n / elapsed if elapsed > 0 else 0.0
            remaining = (total - n) / rate if rate > 0 else 0.0
            eta_str = _fmt_seconds(remaining)
            time_part = f"[{elapsed_str}<{eta_str}]"
        else:
            time_part = f"[{elapsed_str}]"

        prefix = f"{self._label} " if self._label else ""
        if self._is_int_total():
            return f"{prefix}%v/%m {time_part}"
        return f"{prefix}{time_part}"

    def set_description(self, description: str) -> None:
        self._label = description
        if self._disabled:
            return
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def update(self, value: ProgressValue = 1) -> None:
        self._n = self._n + value
        if self._disabled:
            return
        self._bridge.update_requested.emit(self._bar, float(value))
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def reset(self) -> None:
        import time

        self._n = 0
        self._start_time = time.monotonic()
        if self._disabled:
            return
        self._bridge.set_value_requested.emit(self._bar, 0)
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def refresh(self) -> None:
        pass  # Qt repaints automatically

    def close(self) -> None:
        if self._disabled:
            return
        if not self._leave:
            self._bridge.pop_requested.emit(self._bar)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        if self._disabled:
            return
        self._bridge.set_max_requested.emit(self._bar, value)

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._label


class QtProgressBarFactory:
    """Callable factory; pass to use_pbar_factory() or Runner.start_run()."""

    def __init__(self, stack: "ProgressStack") -> None:
        self._bridge = _StackBridge(stack)

    def __call__(self, *args: Any, **kwargs: Any) -> QtProgressBar:
        # tqdm-compatible: positional args may be (iterable, desc, total, ...)
        label = kwargs.pop("desc", "") or (args[1] if len(args) > 1 else "")
        total = kwargs.pop("total", None) or (args[2] if len(args) > 2 else None)
        leave = kwargs.pop("leave", True)
        disabled = bool(kwargs.pop("disable", False))
        return QtProgressBar(
            self._bridge,
            label=str(label) if label else "",
            total=total,
            leave=leave,
            disabled=disabled,
        )

    def set_progress_callback(
        self, cb: Callable[[], None], interval: int = 10
    ) -> None:
        """Delegate to the bridge's step callback mechanism."""
        self._bridge.set_progress_callback(cb, interval)

    def get_all_snapshots(self) -> tuple[RunProgressSnapshot, ...]:
        """Return snapshots of all active progress bars (main-thread only)."""
        return self._bridge.get_all_snapshots()
