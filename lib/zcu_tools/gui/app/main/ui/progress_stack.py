"""Reusable compact progress bar stack widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.pbar_host import ProgressBarModel


class ProgressStack(QWidget):
    """Compact progress bar panel that only occupies space for active bars.

    Bars are added to the layout on push() and removed on pop()/reset_all(),
    so the widget has zero height when idle and grows only as bars are pushed.
    The anti-jitter strategy: bars are reused from a pool so Qt does not
    repeatedly allocate/free widgets; only the layout insertion/removal happens.
    """

    MAX_LAYERS = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._pool: list[QProgressBar] = [
            QProgressBar() for _ in range(self.MAX_LAYERS)
        ]
        self._active: list[QProgressBar] = []

    def push(self, label: str = "", total: int = 0) -> QProgressBar:
        if self._pool:
            bar = self._pool.pop()
        else:
            bar = self._active[-1]  # reuse innermost when all slots busy
            return bar
        bar.setFormat(f"{label} %v/%m" if label else "%v/%m")
        bar.setMaximum(total)
        bar.setValue(0)
        self._layout.insertWidget(0, bar)
        bar.show()
        self._active.append(bar)
        return bar

    def pop(self, bar: QProgressBar) -> None:
        if bar in self._active:
            self._active.remove(bar)
            self._layout.removeWidget(bar)
            bar.hide()
            bar.setParent(self)
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._pool.append(bar)

    def reset_all(self) -> None:
        """Remove all active bars (called when a run ends)."""
        for bar in list(self._active):
            self._layout.removeWidget(bar)
            bar.hide()
            bar.setParent(self)
            bar.setValue(0)
            bar.setFormat("%v/%m")
        self._pool.extend(self._active)
        self._active.clear()

    def render_models(self, models: tuple["ProgressBarModel", ...]) -> None:
        """Replace visible bars with the service-owned live bar models (the View
        calls this from its ProgressService progress listener). Derived values
        are read live off each model (the SSOT) — the widget computes nothing.

        When the *number* of bars is unchanged (the common case: a live bar
        advancing, or its total being reset between sweep points), the existing
        QProgressBar widgets are updated in place — no layout remove/re-add — so
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

    @staticmethod
    def _apply_model(bar: QProgressBar, model: "ProgressBarModel") -> None:
        bar.setMaximum(model.qt_maximum())
        bar.setFormat(model.format())
        bar.setValue(model.qt_value())
