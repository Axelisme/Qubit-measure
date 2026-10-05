"""Disposable Qt brush controls and generation-checked TwoTone projections."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from numpy.typing import NDArray
from qtpy import QtCore, QtWidgets

from zcu_tools.analysis.fluxdep.stroke import (
    BrushMode,
    BrushPoint,
    BrushStroke,
    BrushTool,
)
from zcu_tools.analysis.fluxdep.twotone import (
    TwoTonePickResult,
    TwoTonePickState,
    TwoTonePickView,
    TwoToneSettings,
    analyze_twotone_pick,
    project_twotone_pick,
)
from zcu_tools.gui.app.fluxdep.interactive import TwoTonePickContext
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.plotting.fluxdep.twotone import TwoTonePickPlot

from .base import InteractiveMplWidget

logger = logging.getLogger(__name__)

TwoTonePreviewSubmitter = Callable[
    [
        Callable[[], TwoTonePickView],
        Callable[[TwoTonePickView], None],
        Callable[[Exception], None],
    ],
    None,
]


class FindPointsWidget(InteractiveMplWidget):
    """Controls and pointer preview of one app-owned TwoTone selection.

    Session is the sole authoritative state. Worker results are only derived
    views. Finish emits a request to the app owner, which recomputes exact state.
    """

    def __init__(
        self,
        context: TwoTonePickContext,
        *,
        submit_preview: TwoTonePreviewSubmitter | None = None,
    ) -> None:
        """Attach open context and start an 80ms-debounced derived preview.

        None creates an owned BackgroundRunner. An injected submit_preview must
        deliver on owner and its caller owns external worker cleanup. Controls
        use shared Actions/Undo. Failed Actions reproject committed controls
        before showing their error in the Status label; never coerce conflicting
        detector settings. Closed context raises FailedPreconditionError.
        No spectrum ownership or domain cancellation is transferred.
        """
        context.session.ensure_input_open()
        super().__init__()
        self._context = context
        self._detached = False
        self._generation = 0
        self._view: TwoTonePickView | None = None
        self._observed = context.session.snapshot()
        self._previous: TwoTonePickState | None = None
        self._background = np.zeros(
            context.plugin.inputs.spectrum.signals.shape, dtype=np.float64
        )
        self._vertices: list[BrushPoint] = []
        self._gesture_width = 0.0
        self._gesture_mode: BrushMode = "select"
        self._plot = TwoTonePickPlot(self.figure, context.plugin.inputs)
        (self._pointer,) = self.figure.axes[0].plot(
            [], [], linestyle="--", linewidth=1, color="#007e91", zorder=7
        )
        self._runner = BackgroundRunner(self) if submit_preview is None else None
        self._submit_preview = (
            submit_preview if submit_preview is not None else self._submit_owned
        )
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._start_preview)
        self._threshold = self._slider("Threshold", 1000, 20000)
        self._sigma = self._slider("Smooth", 0, 5000)
        self._width = self._slider("Brush width", 0, 100)
        self._method = self._combo("Smooth method", ("Wavelet", "Gaussian"))
        self._mode = self._combo("Operation", ("Select", "Erase"))
        self._threshold.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_settings.execute(
                    context.session, TwoToneSettings(threshold=value / 1000)
                )
            )
        )
        self._sigma.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_settings.execute(
                    context.session, TwoToneSettings(sigma=value / 1000)
                )
            )
        )
        self._width.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_tool.execute(
                    context.session, BrushTool(width=value / 1000)
                )
            )
        )
        self._method.currentTextChanged.connect(
            lambda text: self._execute(
                lambda: context.plugin.set_settings.execute(
                    context.session,
                    TwoToneSettings(
                        smooth_method="wavelet" if text == "Wavelet" else "gaussian"
                    ),
                )
            )
        )
        self._mode.currentTextChanged.connect(
            lambda text: self._execute(
                lambda: context.plugin.set_tool.execute(
                    context.session,
                    BrushTool(mode="select" if text == "Select" else "erase"),
                )
            )
        )
        self._undo = self._button("Undo", context.session.undo)
        self._button(
            "Clear", lambda: context.plugin.clear.execute(context.session, None)
        )
        self._button(
            "Perform on all",
            lambda: context.plugin.perform_on_all.execute(context.session, None),
        )
        self._show_mask = self._checkbox("Show mask", checked=False)
        self._show_origin = self._checkbox("Show origin", checked=True)
        self._show_changes = self._checkbox("Show changes", checked=True)
        self._status = QtWidgets.QLabel()
        self._status.setAccessibleName("Status")
        self._status.setWordWrap(True)
        self.controls_layout.addWidget(self._status)
        self._finish = self.add_finish_button()
        self._finish.clicked.disconnect()
        self._finish.clicked.connect(self._request_finish)
        self._unsubscribe = context.session.subscribe(self._show_committed)
        self._show_committed()

    def update_points(self) -> None:
        """Invalidate older views and schedule latest detached state off-main.

        Pending scatter/cache is cleared. Latest failure remains visible and
        preview_view returns None. Old/closed/detached completions are discarded.
        This presentation update never commits or changes Undo.
        """
        self._generation += 1
        self._view = None
        self._timer.stop()
        if not self._input_open():
            return
        self._render_pending()
        self._status.setText("Preview pending")
        self._timer.start()

    def preview_view(self) -> TwoTonePickView | None:
        """Return detached latest settled view or None while pending/closed/detached.

        Never recompute inline or expose a stale generation or mutable cache.
        """
        if not self._input_open():
            return None
        return deepcopy(self._view)

    def on_press(self, event: MouseEvent) -> None:
        """Start a left-button pointer preview without committing selection."""
        if event.button != MouseButton.LEFT or not self._input_open():
            return
        point = self._point(event)
        if point is None:
            return
        state = self._context.session.snapshot()
        self._gesture_width, self._gesture_mode = state.width, state.mode
        self._vertices = [point]
        self._render_pointer()

    def on_move(self, event: MouseEvent) -> None:
        """Append finite in-axis vertices while a gesture is active; no commit."""
        if not self._vertices or not self._input_open():
            return
        point = self._point(event)
        if point is not None:
            self._vertices.append(point)
            self._render_pointer()

    def on_release(self, event: MouseEvent) -> None:
        """Commit one complete gesture on latest Session, then clear preview.

        Out-of-axis release uses the last valid vertex. Width/mode are captured
        on press; invalid/closed input is shown without committing. Non-left or
        inactive gestures do not submit.
        """
        if event.button != MouseButton.LEFT or not self._vertices:
            return
        point = self._point(event)
        if point is not None:
            self._vertices.append(point)
        payload = BrushStroke(
            tuple(self._vertices), self._gesture_width, self._gesture_mode
        )
        self._vertices.clear()
        self._render_pointer()
        self._execute(
            lambda: self._context.plugin.stroke.execute(self._context.session, payload)
        )

    def get_result(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Synchronously derive sorted native device/GHz points from exact state.

        Never read preview cache or controls; propagate numerical failures.
        """
        result = analyze_twotone_pick(
            self._context.plugin.inputs, self._context.session.snapshot()
        )
        return result.dev_values, result.freqs

    def quiesce(self) -> None:
        """Idempotently detach/invalidate, stop debounce and join owned runner.

        Unsubscribe and discard late completions without closing Session.
        External injected submitter cleanup remains its caller's responsibility.
        """
        if self._detached:
            return
        self._detached = True
        self._generation += 1
        self._timer.stop()
        self._unsubscribe()
        self._view = None
        self._vertices.clear()
        self._render_pointer()
        self._render_pending()
        for control in self.findChildren(QtWidgets.QWidget):
            if control is not self.canvas:
                control.setEnabled(False)
        if self._runner is not None and not self._runner.quiesce():
            raise RuntimeError("TwoTone preview runner did not quiesce")

    def teardown(self) -> None:
        """Idempotently quiesce presentation without cancelling domain input."""
        self.quiesce()

    def _slider(self, name: str, lower: int, upper: int) -> QtWidgets.QSlider:
        self.controls_layout.addWidget(QtWidgets.QLabel(name))
        control = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        control.setAccessibleName(name)
        control.setRange(lower, upper)
        self.controls_layout.addWidget(control)
        return control

    def _combo(self, name: str, choices: tuple[str, ...]) -> QtWidgets.QComboBox:
        self.controls_layout.addWidget(QtWidgets.QLabel(name))
        control = QtWidgets.QComboBox()
        control.setAccessibleName(name)
        control.addItems(list(choices))
        self.controls_layout.addWidget(control)
        return control

    def _button(
        self, name: str, operation: Callable[[], object]
    ) -> QtWidgets.QPushButton:
        control = QtWidgets.QPushButton(name)
        control.clicked.connect(lambda: self._execute(operation))
        self.controls_layout.addWidget(control)
        return control

    def _checkbox(self, name: str, *, checked: bool) -> QtWidgets.QCheckBox:
        control = QtWidgets.QCheckBox(name)
        control.setChecked(checked)
        control.toggled.connect(self._render)
        self.controls_layout.addWidget(control)
        return control

    def _input_open(self) -> bool:
        if self._detached:
            return False
        try:
            self._context.session.ensure_input_open()
        except FailedPreconditionError:
            self._generation += 1
            self._timer.stop()
            self._view = None
            self._vertices.clear()
            self._render_pointer()
            self._render_pending()
            for control in (
                self._threshold,
                self._sigma,
                self._width,
                self._method,
                self._mode,
                self._undo,
                self._finish,
            ):
                control.setEnabled(False)
            return False
        return True

    def _execute(self, operation: Callable[[], object]) -> None:
        if not self._input_open():
            return
        try:
            operation()
        except (InvalidInputError, FailedPreconditionError) as exc:
            self._reproject_controls()
            self._status.setText(str(exc))
        except Exception as exc:
            # Isolate Qt callbacks; committed state and visible failure survive.
            logger.exception("TwoTone control failed")
            self._reproject_controls()
            self._status.setText(str(exc))

    def _reproject_controls(self) -> None:
        state = self._context.session.snapshot()
        for slider, value in (
            (self._threshold, state.threshold),
            (self._sigma, state.sigma),
            (self._width, state.width),
        ):
            blocked = slider.blockSignals(True)
            slider.setValue(round(value * 1000))
            slider.blockSignals(blocked)
        for combo, text in (
            (
                self._method,
                "Wavelet" if state.smooth_method == "wavelet" else "Gaussian",
            ),
            (self._mode, "Select" if state.mode == "select" else "Erase"),
        ):
            blocked = combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(blocked)
        self._undo.setEnabled(self._context.session.can_undo())

    def _show_committed(self) -> None:
        if not self._input_open():
            return
        state = self._context.session.snapshot()
        if not _same_analysis(state, self._observed):
            self._previous = self._observed
        self._observed = state
        self._reproject_controls()
        self.update_points()

    def _submit_owned(
        self,
        compute: Callable[[], TwoTonePickView],
        on_done: Callable[[TwoTonePickView], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._runner is None:
            raise RuntimeError("owned TwoTone preview runner is unavailable")
        self._runner.submit(compute, on_done=on_done, on_error=on_error)

    def _start_preview(self) -> None:
        if not self._input_open():
            return
        generation = self._generation
        state = self._context.session.snapshot()
        previous = deepcopy(self._previous)
        inputs = self._context.plugin.inputs

        def done(view: TwoTonePickView) -> None:
            if generation != self._generation or not self._input_open():
                return
            self._view = deepcopy(view)
            self._background = view.result.real_signals.copy()
            self._render()
            self._status.setText(
                f"Points: {view.result.dev_values.size}, added: {len(view.added_points)}, "
                f"removed: {len(view.removed_points)}\n"
                f"Mask added: {view.mask_added}, removed: {view.mask_removed}"
            )

        def error(exc: Exception) -> None:
            if generation != self._generation or not self._input_open():
                return
            self._view = None
            self._render_pending()
            self._status.setText(str(exc))

        try:
            self._submit_preview(
                lambda: project_twotone_pick(inputs, state, previous=previous),
                done,
                error,
            )
        except Exception as exc:
            # Submission itself is a preview failure, not a domain mutation.
            logger.exception("TwoTone preview submission failed")
            error(exc)

    def _render(self) -> None:
        if not self._input_open():
            return
        if self._view is None:
            self._render_pending()
            return
        self._plot.show_state(
            self._view,
            show_mask=self._show_mask.isChecked(),
            show_origin=self._show_origin.isChecked(),
            show_changes=self._show_changes.isChecked(),
        )
        self.redraw()

    def _render_pending(self) -> None:
        empty = np.empty(0, dtype=np.float64)
        points = np.empty((0, 2), dtype=np.float64)
        pending = TwoTonePickView(
            self._observed,
            TwoTonePickResult(self._background, empty, empty),
            points,
            points,
            0,
            0,
        )
        self._plot.show_state(
            pending,
            show_mask=self._show_mask.isChecked(),
            show_origin=self._show_origin.isChecked(),
        )
        self.redraw()

    def _point(self, event: MouseEvent) -> BrushPoint | None:
        if (
            event.inaxes is not self.figure.axes[0]
            or event.xdata is None
            or event.ydata is None
        ):
            return None
        if not np.isfinite(event.xdata) or not np.isfinite(event.ydata):
            return None
        return BrushPoint(float(event.xdata), float(event.ydata))

    def _render_pointer(self) -> None:
        self._pointer.set_data(
            [p.x for p in self._vertices], [p.y for p in self._vertices]
        )
        self.redraw()

    def _request_finish(self) -> None:
        if self._input_open():
            self.finished.emit()


def _same_analysis(first: TwoTonePickState, second: TwoTonePickState) -> bool:
    if not (
        np.array_equal(first.mask, second.mask)
        and first.threshold == second.threshold
        and first.sigma == second.sigma
        and first.smooth_method == second.smooth_method
    ):
        return False
    a, b = first.last_change, second.last_change
    if a is None or b is None:
        return a is b
    return (
        np.array_equal(a.mask, b.mask)
        and a.threshold == b.threshold
        and a.sigma == b.sigma
        and a.smooth_method == b.smooth_method
        and a.vertices == b.vertices
        and a.width == b.width
    )
