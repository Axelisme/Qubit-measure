"""One-tone Qt controls and disposable presentation of an app-owned Session."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from qtpy import QtCore, QtWidgets

from zcu_tools.analysis.fluxdep.onetone import analyze_onetone_pick
from zcu_tools.gui.app.fluxdep.interactive import OneTonePickContext
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.plotting.fluxdep.onetone import OneTonePickPlot

from .base import InteractiveMplWidget

logger = logging.getLogger(__name__)


class OneToneWidget(InteractiveMplWidget):
    """Present one committed threshold/peak Session with Undo and Finish requests.

    Controls use context.plugin.set_threshold and Session.undo. Finish is a
    request to the app owner; get_result never reads artist or slider caches.
    """

    def __init__(self, context: OneTonePickContext) -> None:
        """Attach an open context without taking domain ownership.

        Teardown detaches only presentation. The owner controls cancellation,
        spectrum invalidation and terminal publication. Closed input raises
        FailedPreconditionError before attachment.
        """
        context.session.ensure_input_open()
        super().__init__()
        self._context = context
        self._detached = False
        self._plot = OneTonePickPlot(
            self.figure,
            context.plugin.inputs,
            flux_half=context.flux_half,
            flux_int=context.flux_int,
        )
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(50)
        self._redraw_timer.timeout.connect(self.redraw)
        self.controls_layout.addWidget(QtWidgets.QLabel("Threshold"))
        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 500)
        self._threshold_slider.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_threshold.execute(
                    context.session, value / 100
                )
            )
        )
        self.controls_layout.addWidget(self._threshold_slider)
        self._undo = QtWidgets.QPushButton("Undo")
        self._undo.clicked.connect(lambda: self._execute(context.session.undo))
        self.controls_layout.addWidget(self._undo)
        self._info = QtWidgets.QLabel()
        self._info.setWordWrap(True)
        self.controls_layout.addWidget(self._info)
        self._finish = self.add_finish_button()
        self._finish.clicked.disconnect()
        self._finish.clicked.connect(self._request_finish)
        self._unsubscribe = context.session.subscribe(self._show_committed)
        self._show_committed()

    def _input_open(self) -> bool:
        if self._detached:
            return False
        try:
            self._context.session.ensure_input_open()
        except FailedPreconditionError:
            for control in (self._threshold_slider, self._undo, self._finish):
                control.setEnabled(False)
            return False
        return True

    def _execute(self, operation: Callable[[], object]) -> None:
        if not self._input_open():
            return
        try:
            operation()
        except (InvalidInputError, FailedPreconditionError) as exc:
            self._show_committed()
            self._info.setText(str(exc))
        except Exception as exc:
            # Isolate Qt callback failures while preserving a visible error.
            logger.exception("one-tone control failed")
            self._show_committed()
            self._info.setText(str(exc))

    def _show_committed(self) -> None:
        if self._detached:
            return
        state = self._context.session.snapshot()
        self._plot.show_state(state)
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(round(state.threshold * 100))
        self._threshold_slider.blockSignals(False)
        if self._input_open():
            self._undo.setEnabled(self._context.session.can_undo())
        self._info.setText(
            f"Threshold: {state.threshold:g}, points: {len(state.peak_indices)}"
        )
        self._redraw_timer.start()

    def _request_finish(self) -> None:
        if self._input_open():
            self.finished.emit()

    def get_result(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return committed native device/GHz points, including an empty result."""
        result = analyze_onetone_pick(
            self._context.plugin.inputs, self._context.session.snapshot()
        )
        return result.dev_values, result.freqs

    def teardown(self) -> None:
        """Idempotently stop redraw and detach controls without closing input."""
        if self._detached:
            return
        self._detached = True
        self._redraw_timer.stop()
        self._unsubscribe()
        for control in (self._threshold_slider, self._undo, self._finish):
            control.setEnabled(False)
