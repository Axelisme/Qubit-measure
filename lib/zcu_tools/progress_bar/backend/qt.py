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
    from zcu_tools.gui.ui.main_window import _ProgressStack


class _StackBridge(QObject):
    """Lives in the main thread; receives signals and forwards to _ProgressStack."""

    push_requested: Signal = Signal(str, int)  # label, total → emitted by worker
    pop_requested: Signal = Signal(object)  # QProgressBar → emitted by worker
    update_requested: Signal = Signal(object, int)  # QProgressBar, delta
    set_value_requested: Signal = Signal(object, int)  # QProgressBar, value
    set_max_requested: Signal = Signal(object, int)  # QProgressBar, maximum
    set_format_requested: Signal = Signal(object, str)  # QProgressBar, fmt

    def __init__(self, stack: "_ProgressStack") -> None:
        super().__init__()
        self._stack = stack
        # keep a mapping worker_id → QProgressBar so push_requested can store the result
        self._pending: dict[int, Any] = {}

        self.push_requested.connect(self._on_push)
        self.pop_requested.connect(self._on_pop)
        self.update_requested.connect(self._on_update)
        self.set_value_requested.connect(self._on_set_value)
        self.set_max_requested.connect(self._on_set_max)
        self.set_format_requested.connect(self._on_set_format)

    def _on_push(self, label: str, total: int) -> None:
        # Result stored in _pending keyed by label+total; QtProgressBar polls it.
        bar = self._stack.push(label, total)
        key = id(bar)
        self._pending[key] = bar

    def _on_pop(self, bar: object) -> None:
        self._stack.pop(bar)  # type: ignore[arg-type]

    def _on_update(self, bar: object, delta: int) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            bar.setValue(bar.value() + delta)

    def _on_set_value(self, bar: object, value: int) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            bar.setValue(value)

    def _on_set_max(self, bar: object, maximum: int) -> None:
        from qtpy.QtWidgets import QProgressBar  # type: ignore[attr-defined]

        if isinstance(bar, QProgressBar):
            bar.setMaximum(maximum)

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


class QtProgressBar(BaseProgressBar):
    """BaseProgressBar that drives a QProgressBar on _ProgressStack via signals."""

    def __init__(
        self,
        bridge: "_StackBridge",
        label: str = "",
        total: Optional[ProgressTotal] = None,
        **_kwargs: Any,
    ) -> None:
        self._bridge = bridge
        self._label = label
        self._total: ProgressTotal = total
        self._n: ProgressValue = 0
        # emit push; block briefly until main thread creates the QProgressBar
        bridge.push_requested.emit(label, int(total) if total is not None else 0)
        self._bar = bridge.pop_pending()

    # ------------------------------------------------------------------

    def set_description(self, description: str) -> None:
        self._label = description
        fmt = f"{description} %v/%m" if description else "%v/%m"
        self._bridge.set_format_requested.emit(self._bar, fmt)

    def update(self, value: ProgressValue = 1) -> None:
        self._n = self._n + value
        self._bridge.update_requested.emit(self._bar, int(value))

    def reset(self) -> None:
        self._n = 0
        self._bridge.set_value_requested.emit(self._bar, 0)

    def refresh(self) -> None:
        pass  # Qt repaints automatically

    def close(self) -> None:
        self._bridge.pop_requested.emit(self._bar)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        self._bridge.set_max_requested.emit(
            self._bar, int(value) if value is not None else 0
        )

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._label


class QtProgressBarFactory:
    """Callable factory; pass to use_pbar_factory() or Runner.start_run()."""

    def __init__(self, stack: "_ProgressStack") -> None:
        self._bridge = _StackBridge(stack)

    def __call__(self, *args: Any, **kwargs: Any) -> QtProgressBar:
        # tqdm-compatible: positional args may be (iterable, desc, total, ...)
        # We only care about desc/total/label keywords.
        label = kwargs.pop("desc", "") or (args[1] if len(args) > 1 else "")
        total = kwargs.pop("total", None) or (args[2] if len(args) > 2 else None)
        return QtProgressBar(
            self._bridge, label=str(label) if label else "", total=total
        )
