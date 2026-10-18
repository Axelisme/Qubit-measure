"""Measure-only Qt frontend: click-follow preview and click-to-place commit."""

from __future__ import annotations

import logging
from collections.abc import Callable
from math import isfinite
from typing import Any, cast

from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QHideEvent, QKeyEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import FluxPickState
from zcu_tools.gui.app.measure.flux_pick_plugin import FluxPickPlugin
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.plotting.plots import Plots

logger = logging.getLogger(__name__)


class FluxPickFrontend(InteractiveFrontend):
    """Render committed picker state with disposable click-follow line previews.

    plugin supplies spectrum/actions; session is an open owner-loop Session;
    env supplies frontend services. request_finish/request_cancel ask the owning
    analysis service to settle, returning terminal success. parent is the Qt
    widget parent. Construction subscribes to session/alignment and creates a
    Qt canvas; use on the Qt owner thread and call teardown on retirement.
    Preview artists never supply terminal analysis data."""

    def __init__(
        self,
        plugin: FluxPickPlugin,
        session: Session[FluxPickState],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        *,
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
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._read_only = False
        self._alignment_busy = False

        controls = QWidget(self)
        buttons = QVBoxLayout(controls)
        controls.setFixedWidth(200)
        self._info = QLabel(self._picker.info_text())
        make_label_selectable(self._info)
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
        for widget in (self._conjugate, self._align, self._swap, self._info):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        buttons.addWidget(self._done)
        buttons.addWidget(self._cancel)
        root = QHBoxLayout(self)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(controls)
        self._canvas.installEventFilter(self)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_move)
        self._unsubscribe = session.subscribe(self._on_committed)
        self._unsubscribe_alignment = plugin.subscribe_alignment(self._on_alignment)
        self._canvas.draw_idle()

    def set_read_only(self, *, read_only: bool) -> None:
        """Set command/pointer editing permission without blocking figure viewing.

        read_only=True discards preview and disables commands; False re-enables
        only non-retired controls, retaining alignment's busy gate."""
        self._read_only = read_only
        if read_only:
            self.cancel_preview()
        for widget in (self._conjugate, self._swap, self._done, self._cancel):
            widget.setEnabled(not read_only and not self._retired)
        self._align.setEnabled(
            not read_only and not self._retired and not self._alignment_busy
        )

    @property
    def figure(self) -> Figure:
        """Return the borrowed preview Figure, not the terminal result figure."""
        return self._figure

    @property
    def preview_active(self) -> bool:
        """Report an uncommitted pointer preview on a non-retired frontend."""
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
        """Restore latest committed session state; do nothing after teardown."""
        if not self._retired:
            self._show_committed(self._session.snapshot())

    def _on_press(self, event: MouseEvent) -> None:
        if self._read_only or self._retired or event.button != MouseButton.LEFT:
            return
        role = self._picker.selected_role
        if role is None:
            if self._picker.is_main_axes(event.inaxes):
                self._picker.on_press(event.xdata)
                if self._picker.selected_role is not None:
                    self._canvas.setFocus(Qt.FocusReason.MouseFocusReason)
            return
        x = event.xdata
        if (
            not self._picker.is_main_axes(event.inaxes)
            or x is None
            or event.ydata is None
            or not isfinite(x)
        ):
            self.cancel_preview()
            return
        # Recalculate against the latest committed snapshot; preview artists
        # never supply the replacement state.
        try:
            committed = self._plugin.actions.move.execute(self._session, (role, x))
        except InvalidInputError as exc:
            self.cancel_preview()
            self._info.setText(str(exc))
            return
        position = committed.flux_half if role == "half" else committed.flux_int
        self._picker.show_loss(role, position, event.ydata)
        self._repaint()

    def _on_move(self, event: MouseEvent) -> None:
        if (
            self._read_only
            or self._retired
            or not self._picker.is_main_axes(event.inaxes)
        ):
            return
        x = event.xdata
        role = self._picker.selected_role
        if role is not None and x is not None and isfinite(x):
            try:
                self._plugin.actions.move.calculate(self._committed, (role, x))
            except InvalidInputError:
                return
            self._picker.on_move(x)
            self._preview_active = True
            self._repaint()

    # tolerated: Qt toggled(bool) delivers the checked state positionally.
    def _set_conjugate(self, enabled: bool) -> None:  # noqa: FBT001
        if not self._read_only and not self._retired:
            self._plugin.actions.conjugate.execute(self._session, enabled)

    def _swap_lines(self) -> None:
        if not self._read_only and not self._retired:
            self._plugin.actions.swap.execute(self._session, None)

    # tolerated: Shared alignment subscriptions deliver positional busy/error values.
    def _on_alignment(self, busy: bool, error: str | None) -> None:  # noqa: FBT001
        if self._retired:
            return
        self._alignment_busy = busy
        self._align.setEnabled(not self._read_only and not busy)
        if error is not None:
            self._info.setText(error)

    def _auto_align(self) -> None:
        if self._read_only or self._retired:
            return
        try:
            self._plugin.start_alignment(self._session)
        except FailedPreconditionError as exc:
            self._info.setText(str(exc))
        except Exception as exc:
            logger.exception("Auto Align submission failed")
            self._info.setText(str(exc))

    def _finish(self) -> None:
        if self._read_only or self._retired:
            return
        self.cancel_preview()
        try:
            terminal = self._request_finish()
        except FailedPreconditionError as exc:
            self._info.setText(str(exc))
            return
        if terminal:
            self.teardown()

    def _cancel_analysis(self) -> None:
        if not self._read_only and not self._retired:
            self.cancel_preview()
            self._request_cancel()
            self.teardown()

    def teardown(self) -> None:
        """Idempotently retire controls/subscriptions and discard pointer preview.

        Does not finish/cancel the service-owned session or destroy saved figures."""
        if self._retired:
            return
        self._retired = True
        self._canvas.removeEventFilter(self)
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

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Discard preview on canvas focus loss/Esc; delegate other Qt events.

        a0 is the watched object and a1 the Qt event, either may be None. Returns
        True only when Escape is consumed. Must run on the Qt owner thread."""
        if a0 is self._canvas and a1 is not None:
            if a1.type() == QEvent.Type.FocusOut:
                self.cancel_preview()
            elif a1.type() == QEvent.Type.KeyPress:
                key_event = cast(QKeyEvent, a1)
                if key_event.key() == Qt.Key.Key_Escape:
                    self.cancel_preview()
                    return True
        return super().eventFilter(a0, a1)

    def hideEvent(self, a0: QHideEvent | None) -> None:
        """Discard local preview and forward the optional Qt hide event to QWidget."""
        self.cancel_preview()
        super().hideEvent(a0)


def make_flux_pick_frontend(
    plugin: PluginDefinition[Any, Any],
    session: Session[Any],
    env: InteractiveFrontendEnv,
    request_finish: Callable[[], bool],
    request_cancel: Callable[[], bool],
    *,
    plots: Plots,
) -> InteractiveFrontend:
    """Build the measure frontend for a FluxPickPlugin and its open session.

    env supplies frontend services; request_finish/request_cancel settle the
    owning analysis operation. plots is that operation's terminal figure owner,
    not the preview owner. A different plugin raises TypeError. Returns the Qt
    frontend, whose caller owns teardown; this does not open/close the session."""
    if not isinstance(plugin, FluxPickPlugin):
        raise TypeError("flux-pick adapter requires a FluxPickPlugin")
    # The preview is disposable; the plugin writes final figures into these plots.
    del plots
    return FluxPickFrontend(
        plugin,
        session,
        env,
        request_finish,
        request_cancel,
    )
