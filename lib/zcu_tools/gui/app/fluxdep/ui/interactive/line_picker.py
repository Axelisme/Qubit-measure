"""Session-backed fluxdep line picking, with disposable pointer preview."""

from __future__ import annotations

import logging
from collections.abc import Callable

from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy import QtCore, QtGui, QtWidgets

from zcu_tools.analysis.fluxdep import (
    TwoLinePicker,
    find_best_mirror_position,
    fold_initial_lines,
)
from zcu_tools.gui.app.fluxdep.interactive import LinePickContext
from zcu_tools.gui.app.fluxdep.ui.interactive.base import InteractiveMplWidget
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.plotting.fluxdep.pick import configure_flux_pick_axes

__all__ = ["LinePickerWidget", "find_best_mirror_position", "fold_initial_lines"]
logger = logging.getLogger(__name__)


class LinePickerWidget(InteractiveMplWidget):
    """Present one owner's live context, without owning committed line state.

    Controls use context.plugin actions and context.session undo. The finished
    signal requests owner Finish; get_result is only a committed-state read.
    Invalid moves restore the snapshot and display their error, not a commit.
    """

    def __init__(self, context: LinePickContext) -> None:
        """Attach to an open context and capture its immutable rendering inputs.

        Closing the context rejects input. Teardown detaches only presentation;
        the owner may mount another view of the same session.
        """
        context.session.ensure_input_open()
        super().__init__()
        self._context = context
        self._detached = False
        inputs = context.plugin.inputs
        state = context.session.snapshot()
        self._picker = TwoLinePicker(
            self.figure,
            inputs.signals,
            inputs.dev_values,
            inputs.freqs,
            flux_half=state.flux_half,
            flux_int=state.flux_int,
            force_magnitude=state.magnitude_only,
        )
        configure_flux_pick_axes(self.figure)
        self._conjugate = QtWidgets.QCheckBox("Conjugate Line")
        self._conjugate.toggled.connect(
            lambda enabled: self._execute(
                lambda: context.plugin.actions.conjugate.execute(
                    context.session, enabled
                )
            )
        )
        self.controls_layout.addWidget(self._conjugate)
        self._swap = self._add_button(
            "Swap Lines",
            lambda: context.plugin.actions.swap.execute(context.session, None),
        )
        self._align = self._add_button(
            "Auto Align", lambda: context.plugin.start_alignment(context.session)
        )
        self._undo = self._add_button("Undo", context.session.undo)
        self._info = QtWidgets.QLabel()
        self._info.setWordWrap(True)
        self.controls_layout.addWidget(self._info)
        self._finish = self.add_finish_button()
        # Replace the base's unguarded signal forwarding with an input-gated request.
        self._finish.clicked.disconnect()
        self._finish.clicked.connect(self._request_finish)
        self._unsubscribe = context.session.subscribe(self._show_committed)
        self._unsubscribe_alignment = context.plugin.subscribe_alignment(
            lambda _busy, _error: self._refresh_controls()
        )
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.installEventFilter(self)
        self.installEventFilter(self)
        self._show_committed()

    def _add_button(
        self, label: str, operation: Callable[[], object]
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(lambda: self._execute(operation))
        self.controls_layout.addWidget(button)
        return button

    def _input_open(self) -> bool:
        if self._detached:
            return False
        try:
            self._context.session.ensure_input_open()
        except FailedPreconditionError:
            for control in (
                self._conjugate,
                self._swap,
                self._align,
                self._undo,
                self._finish,
            ):
                control.setEnabled(False)
            return False
        return True

    def _execute(self, operation: Callable[[], object]) -> None:
        if not self._input_open():
            return
        self.cancel_preview()
        try:
            operation()
        except (InvalidInputError, FailedPreconditionError) as exc:
            self._show_committed()
            self._info.setText(str(exc))
        except Exception as exc:
            # Qt input callbacks isolate failures rather than aborting the GUI loop.
            logger.exception("line-picker control failed")
            self._show_committed()
            self._info.setText(str(exc))

    def _request_finish(self) -> None:
        if self._input_open():
            self.cancel_preview()
            self.finished.emit()

    def _refresh_controls(self) -> None:
        if not self._input_open():
            return
        self._undo.setEnabled(self._context.session.can_undo())
        self._align.setEnabled(not self._context.plugin.alignment_busy)
        error = self._context.plugin.info()["alignment_error"]
        self._info.setText(
            str(error) if error is not None else self._picker.info_text()
        )

    def _show_committed(self) -> None:
        if self._detached:
            return
        state = self._context.session.snapshot()
        self._picker.show_state(state)
        self._conjugate.blockSignals(True)
        self._conjugate.setChecked(state.conjugate)
        self._conjugate.blockSignals(False)
        self._refresh_controls()
        self.redraw()

    def on_press(self, event: MouseEvent) -> None:
        """Select a line on the main spectrum axes; do not commit."""
        if (
            self._input_open()
            and event.button == MouseButton.LEFT
            and self._picker.is_main_axes(event.inaxes)
        ):
            self._picker.on_press(event.xdata)

    def on_move(self, event: MouseEvent) -> None:
        """Preview a valid device position while a line is selected."""
        if not self._input_open() or not self._picker.is_main_axes(event.inaxes):
            return
        try:
            self._picker.on_move(event.xdata)
        except ValueError as exc:
            self.cancel_preview()
            self._info.setText(str(exc))
        else:
            self._info.setText(self._picker.info_text())
            self.redraw()

    def on_release(self, event: MouseEvent) -> None:
        """Commit a valid selected-line placement through the shared move action."""
        if not self._input_open():
            return
        role = self._picker.selected_role
        if (
            not self._picker.is_main_axes(event.inaxes)
            or event.xdata is None
            or event.ydata is None
            or event.button != MouseButton.LEFT
        ):
            self.cancel_preview()
            return
        if role is not None:
            position = float(event.xdata)
            self._execute(
                lambda: self._context.plugin.actions.move.execute(
                    self._context.session, (role, position)
                )
            )

    def get_result(self) -> tuple[float, float]:
        """Read committed half/integer device positions, not pointer preview."""
        state = self._context.session.snapshot()
        return state.flux_half, state.flux_int

    def cancel_preview(self) -> None:
        """Restore the latest snapshot on focus loss, hide or Escape."""
        self._show_committed()

    def teardown(self) -> None:
        """Idempotently detach subscriptions and controls, leaving context open."""
        if self._detached:
            return
        self.cancel_preview()
        self._detached = True
        self._unsubscribe()
        self._unsubscribe_alignment()
        self.canvas.removeEventFilter(self)
        self.removeEventFilter(self)
        for control in (
            self._conjugate,
            self._swap,
            self._align,
            self._undo,
            self._finish,
        ):
            control.setEnabled(False)

    def _dispatch_release(self, event: object) -> None:
        # The base filters outside-axes releases, but those must discard gestures.
        if isinstance(event, MouseEvent):
            self.on_release(event)

    def eventFilter(self, a0: QtCore.QObject | None, a1: QtCore.QEvent | None) -> bool:
        """Discard presentation-only gestures on focus loss, hide or Escape."""
        if a1 is not None:
            if a1.type() in (QtCore.QEvent.Type.FocusOut, QtCore.QEvent.Type.Hide):
                self.cancel_preview()
            elif (
                isinstance(a1, QtGui.QKeyEvent) and a1.key() == QtCore.Qt.Key.Key_Escape
            ):
                self.cancel_preview()
                return True
        return super().eventFilter(a0, a1)
