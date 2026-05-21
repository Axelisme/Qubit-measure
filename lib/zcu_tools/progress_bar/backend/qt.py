"""Qt progress bar backend — bridges BaseProgressBar API to _ProgressStack via Qt signals.

Because RunWorker runs on a QThread, widget operations must be dispatched to
the main thread. This module uses Qt signals (emitted from any thread, received
in the main thread) to push/pop/update the _ProgressStack widget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from ..base import BaseProgressBar, ProgressTotal, ProgressValue

if TYPE_CHECKING:
    from zcu_tools.gui.ui.progress_stack import ProgressStack


_SCALE = 1000  # QProgressBar is integer-only; map [0, total] → [0, _SCALE]


class _StackBridge(QObject):
    """Lives in the main thread; receives signals and forwards to ProgressStack.

    All progress values are scaled to [0, _SCALE] integers so that QProgressBar
    (which is integer-only) can represent arbitrary float totals accurately.
    """

    push_requested: Signal = Signal(str, float)  # label, total
    pop_requested: Signal = Signal(object)  # QProgressBar
    update_requested: Signal = Signal(
        object, float
    )  # QProgressBar, delta (original units)
    set_value_requested: Signal = Signal(
        object, float
    )  # QProgressBar, value (original units)
    set_max_requested: Signal = Signal(
        object, float
    )  # QProgressBar, total (original units)
    set_format_requested: Signal = Signal(object, str)  # QProgressBar, fmt

    def __init__(self, stack: "ProgressStack") -> None:
        super().__init__()
        self._stack = stack
        self._pending: dict[int, Any] = {}
        # per-bar scale factor: maps original total → _SCALE
        self._scales: dict[int, float] = {}

        self.push_requested.connect(self._on_push)
        self.pop_requested.connect(self._on_pop)
        self.update_requested.connect(self._on_update)
        self.set_value_requested.connect(self._on_set_value)
        self.set_max_requested.connect(self._on_set_max)
        self.set_format_requested.connect(self._on_set_format)

    def _scale_of(self, bar: Any) -> float:
        return self._scales.get(id(bar), 1.0)

    def _on_push(self, label: str, total: float) -> None:
        bar = self._stack.push(label, _SCALE)
        key = id(bar)
        self._pending[key] = bar
        self._scales[key] = (_SCALE / total) if total > 0 else 1.0

    def _on_pop(self, bar: object) -> None:
        self._scales.pop(id(bar), None)
        self._stack.pop(bar)  # type: ignore[arg-type]

    def _on_update(self, bar: object, delta: float) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            scaled = int(round(delta * self._scale_of(bar)))
            bar.setValue(bar.value() + scaled)

    def _on_set_value(self, bar: object, value: float) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            bar.setValue(int(round(value * self._scale_of(bar))))

    def _on_set_max(self, bar: object, total: float) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            self._scales[id(bar)] = (_SCALE / total) if total > 0 else 1.0
            bar.setMaximum(_SCALE)

    def _on_set_format(self, bar: object, fmt: str) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            bar.setFormat(fmt)

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
        **_kwargs: Any,
    ) -> None:
        import time

        self._bridge = bridge
        self._label = label
        self._total: ProgressTotal = total
        self._leave = leave
        self._n: ProgressValue = 0
        self._start_time: float = time.monotonic()
        bridge.push_requested.emit(label, float(total) if total is not None else 0.0)
        self._bar = bridge.pop_pending()

    # ------------------------------------------------------------------

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
        return f"{prefix}%v/%m {time_part}"

    def set_description(self, description: str) -> None:
        self._label = description
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def update(self, value: ProgressValue = 1) -> None:
        self._n = self._n + value
        self._bridge.update_requested.emit(self._bar, float(value))
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def reset(self) -> None:
        import time

        self._n = 0
        self._start_time = time.monotonic()
        self._bridge.set_value_requested.emit(self._bar, 0)
        self._bridge.set_format_requested.emit(self._bar, self._build_format())

    def refresh(self) -> None:
        pass  # Qt repaints automatically

    def close(self) -> None:
        if not self._leave:
            self._bridge.pop_requested.emit(self._bar)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        self._bridge.set_max_requested.emit(
            self._bar, float(value) if value is not None else 0.0
        )

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
        return QtProgressBar(
            self._bridge,
            label=str(label) if label else "",
            total=total,
            leave=leave,
        )
