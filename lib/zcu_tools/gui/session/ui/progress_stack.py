"""Reusable compact progress bar stack widget."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget  # type: ignore[attr-defined]

from zcu_tools.gui.session.ui.progress_bar import LightweightProgressBar

if TYPE_CHECKING:
    from zcu_tools.gui.session.pbar_host import ProgressBarModel

_BarSnapshot = tuple[int, str, int]
_ProfileCallback = Callable[[str, float, str], None]


class ProgressStack(QWidget):
    """Compact progress bar panel that only occupies space for active bars.

    Bars are added to the layout on push() and removed on pop()/reset_all(),
    so the widget has zero height when idle and grows only as bars are pushed.
    The anti-jitter strategy: bars are reused from a pool so Qt does not
    repeatedly allocate/free widgets; only the layout insertion/removal happens.
    """

    MAX_LAYERS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._pool: list[LightweightProgressBar] = [
            LightweightProgressBar() for _ in range(self.MAX_LAYERS)
        ]
        self._active: list[LightweightProgressBar] = []
        self._snapshots: dict[int, _BarSnapshot] = {}
        self._profile_callback: _ProfileCallback | None = None

    def set_profile_callback(self, callback: _ProfileCallback | None) -> None:
        self._profile_callback = callback

    def _profile(self, label: str, start: float, *, detail: str = "") -> None:
        if self._profile_callback is None:
            return
        self._profile_callback(label, (time.perf_counter() - start) * 1000.0, detail)

    def push(self, label: str = "", total: int = 0) -> LightweightProgressBar:
        start = time.perf_counter()
        if self._pool:
            bar = self._pool.pop()
        else:
            bar = self._active[-1]  # reuse innermost when all slots busy
            self._profile("push_reuse", start)
            return bar
        bar.setFormat(f"{label} %v/%m" if label else "%v/%m")
        bar.setMaximum(total)
        bar.setValue(0)
        self._layout.insertWidget(0, bar)
        bar.show()
        self._active.append(bar)
        self._profile("push", start)
        return bar

    def pop(self, bar: LightweightProgressBar) -> None:
        start = time.perf_counter()
        if bar in self._active:
            self._active.remove(bar)
            self._layout.removeWidget(bar)
            bar.hide()
            bar.setParent(self)
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._snapshots.pop(id(bar), None)
            self._pool.append(bar)
        self._profile("pop", start)

    def reset_all(self) -> None:
        """Remove all active bars (called when a run ends)."""
        start = time.perf_counter()
        for bar in list(self._active):
            self._layout.removeWidget(bar)
            bar.hide()
            bar.setParent(self)
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._snapshots.pop(id(bar), None)
        self._pool.extend(self._active)
        self._active.clear()
        self._profile("reset_all", start)

    def render_models(self, models: tuple[ProgressBarModel, ...]) -> None:
        """Replace visible bars with the service-owned live bar models (the View
        calls this from its ProgressService progress listener). Derived values
        are read live off each model (the SSOT) — the widget computes nothing.

        When the *number* of bars is unchanged (the common case: a live bar
        advancing, or its total being reset between sweep points), the existing
        progress widgets are updated in place — no layout remove/re-add — so
        the bars never flicker out of view. Only a change in bar count rebuilds
        the stack from the pool.
        """
        shown = models[: self.MAX_LAYERS]
        if len(shown) == len(self._active):
            for bar, model in zip(self._active, shown):
                self._apply_model(bar, model)
            return
        self.reset_all()
        for model in shown:
            bar = self.push(total=model.qt_maximum())
            self._apply_model(bar, model)

    def _apply_model(
        self, bar: LightweightProgressBar, model: ProgressBarModel
    ) -> None:
        derive_start = time.perf_counter()
        maximum = model.qt_maximum()
        fmt = model.format()
        value = model.qt_value()
        self._profile("derive", derive_start, detail=f"label={model.label}")
        self._apply_bar_snapshot(
            bar,
            (
                maximum,
                fmt,
                value,
            ),
        )

    def _apply_bar_snapshot(
        self, bar: LightweightProgressBar, snapshot: _BarSnapshot
    ) -> None:
        start = time.perf_counter()
        old = self._snapshots.get(id(bar))
        if old == snapshot:
            self._profile("snapshot_noop", start)
            return
        maximum, fmt, value = snapshot
        if old is None or old[0] != maximum:
            step_start = time.perf_counter()
            bar.setMaximum(maximum)
            self._profile("set_maximum", step_start)
        if old is None or old[1] != fmt:
            step_start = time.perf_counter()
            bar.setFormat(fmt)
            self._profile("set_format", step_start)
        if old is None or old[2] != value:
            step_start = time.perf_counter()
            bar.setValue(value)
            self._profile("set_value", step_start)
        self._snapshots[id(bar)] = snapshot
        self._profile("apply_snapshot", start)
