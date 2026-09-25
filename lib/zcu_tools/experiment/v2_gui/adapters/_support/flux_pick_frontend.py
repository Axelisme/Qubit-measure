"""Measure-only Qt frontend: local line preview, committed actions on release."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, cast

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFocusEvent, QHideEvent, QKeyEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import FluxPickState
from zcu_tools.gui.app.main.interactive import PluginDefinition, Session
from zcu_tools.gui.app.main.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.expected_error import FailedPreconditionError

from .flux_pick_plugin import FluxPickPlugin


class FluxPickFrontend(InteractiveFrontend):
    """Render the latest session snapshot, or a disposable local drag candidate."""

    def __init__(
        self,
        plugin: FluxPickPlugin,
        session: Session[FluxPickState],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[Figure], bool],
        request_cancel: Callable[[], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plugin = plugin
        self._session = session
        self._env = env
        self._request_finish = request_finish
        self._request_cancel = request_cancel
        self._figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvasQTAgg(self._figure)
        state = session.snapshot()
        inputs = plugin.inputs
        self._picker = TwoLinePicker(
            self._figure,
            inputs.signals,
            inputs.dev_values,
            inputs.freqs,
            flux_half=state.flux_half,
            flux_int=state.flux_int,
            force_magnitude=state.magnitude_only,
        )
        self._picker.show_state(state)
        self._committed = state
        self._preview_active = False
        self._retired = False

        controls = QWidget(self)
        buttons = QVBoxLayout(controls)
        controls.setFixedWidth(200)
        self._info = QLabel(self._picker.info_text())
        self._info.setWordWrap(True)
        self._conjugate = QCheckBox("Conjugate Line")
        self._conjugate.setChecked(state.conjugate)
        self._conjugate.toggled.connect(self._set_conjugate)
        self._align = QPushButton("Auto Align")
        self._align.clicked.connect(self._auto_align)
        self._swap = QPushButton("Swap Lines")
        self._swap.clicked.connect(self._swap_lines)
        self._done = QPushButton("Done")
        self._done.clicked.connect(self._finish)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._cancel_analysis)
        for widget in (
            self._conjugate,
            self._align,
            self._swap,
            self._info,
            self._done,
            self._cancel,
        ):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        root = QHBoxLayout(self)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(controls)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_move)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._unsubscribe = session.subscribe(self._on_committed)
        self._unsubscribe_alignment = plugin.subscribe_alignment(self._on_alignment)
        self._canvas.draw_idle()

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return self._preview_active and not self._retired

    def _repaint(self) -> None:
        self._info.setText(self._picker.info_text())
        self._canvas.draw_idle()

    def _show_committed(self, state: FluxPickState) -> None:
        self._picker.show_state(state)
        self._committed = state
        self._preview_active = False
        self._conjugate.blockSignals(True)
        self._conjugate.setChecked(state.conjugate)
        self._conjugate.blockSignals(False)
        self._repaint()

    def _on_committed(self) -> None:
        if not self._retired:
            self._show_committed(self._session.snapshot())

    def cancel_preview(self) -> None:
        if not self._retired:
            self._show_committed(self._session.snapshot())

    def _on_press(self, event: MouseEvent) -> None:
        if not self._retired and self._picker.is_main_axes(event.inaxes):
            self.cancel_preview()
            self._picker.on_press(event.xdata)

    def _on_move(self, event: MouseEvent) -> None:
        if self._retired or not self._picker.is_main_axes(event.inaxes):
            return
        x = event.xdata
        if self._picker.selected_role is not None and x is not None and isfinite(x):
            self._picker.on_move(x)
            self._preview_active = True
            self._repaint()

    def _on_release(self, event: MouseEvent) -> None:
        if self._retired:
            return
        role = self._picker.selected_role
        x = event.xdata
        if (
            role is None
            or not self._picker.is_main_axes(event.inaxes)
            or x is None
            or event.ydata is None
            or not isfinite(x)
        ):
            self.cancel_preview()
            return
        # The Action recalculates on the latest committed snapshot. The picker
        # is just a local artist cache and never supplies the replacement state.
        committed = self._plugin.actions.move.execute(self._session, (role, x))
        position = committed.flux_half if role == "half" else committed.flux_int
        self._picker.show_loss(role, position, event.ydata)
        self._repaint()

    def _set_conjugate(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool)
        if not self._retired:
            self._plugin.actions.conjugate.execute(self._session, enabled)

    def _swap_lines(self) -> None:
        if not self._retired:
            self._plugin.actions.swap.execute(self._session, None)

    def _on_alignment(self, busy: bool, error: str | None) -> None:  # noqa: FBT001 - status subscription
        if self._retired:
            return
        self._align.setEnabled(not busy)
        if error is not None:
            self._info.setText(error)

    def _auto_align(self) -> None:
        if self._retired:
            return
        try:
            self._plugin.start_alignment(self._session)
        except FailedPreconditionError as exc:
            self._info.setText(str(exc))
        except Exception as exc:  # noqa: BLE001 - failed pool submission
            self._info.setText(str(exc))

    def _finish(self) -> None:
        if self._retired:
            return
        self.cancel_preview()
        try:
            terminal = self._request_finish(self._figure)
        except FailedPreconditionError as exc:
            self._info.setText(str(exc))
            return
        if terminal:
            self.teardown()

    def _cancel_analysis(self) -> None:
        if not self._retired:
            self.cancel_preview()
            self._request_cancel()
            self.teardown()

    def teardown(self) -> None:
        if self._retired:
            return
        self._retired = True
        self._unsubscribe()
        self._unsubscribe_alignment()
        self._picker.show_state(self._committed)
        for widget in (
            self._conjugate,
            self._align,
            self._swap,
            self._done,
            self._cancel,
        ):
            widget.setEnabled(False)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is not None and a0.key() == Qt.Key.Key_Escape:
            self.cancel_preview()
            a0.accept()
        else:
            super().keyPressEvent(a0)

    def focusOutEvent(self, a0: QFocusEvent | None) -> None:
        self.cancel_preview()
        super().focusOutEvent(a0)

    def hideEvent(self, a0: QHideEvent | None) -> None:
        self.cancel_preview()
        super().hideEvent(a0)


def make_flux_pick_frontend(
    plugin: PluginDefinition[Any, Any],
    session: Session[Any],
    env: InteractiveFrontendEnv,
    request_finish: Callable[[Figure], bool],
    request_cancel: Callable[[], bool],
) -> InteractiveFrontend:
    if not isinstance(plugin, FluxPickPlugin):
        raise TypeError("flux-pick adapter requires a FluxPickPlugin")
    return FluxPickFrontend(
        plugin,
        cast(Session[FluxPickState], session),
        env,
        request_finish,
        request_cancel,
    )
