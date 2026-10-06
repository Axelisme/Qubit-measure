"""Disposable Qt controls for the app-owned joint-cloud selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy import QtCore, QtWidgets

from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionResult,
    CrossSelectionState,
    CrossSelectionView,
    analyze_cross_selection,
    project_cross_selection,
)
from zcu_tools.analysis.fluxdep.stroke import (
    BrushMode,
    BrushPoint,
    BrushStroke,
    BrushTool,
)
from zcu_tools.gui.app.fluxdep.interactive import CrossSelectionContext
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.plotting.fluxdep.cross_selection import CrossSelectionPlot

from .base import InteractiveMplWidget

logger = logging.getLogger(__name__)

CrossSelectionPreviewSubmitter = Callable[
    [
        Callable[[], CrossSelectionView],
        Callable[[CrossSelectionView], None],
        Callable[[Exception], None],
    ],
    None,
]


class SelectorWidget(InteractiveMplWidget):
    """View of one context, never a second committed selection/filter mask.

    Controls use shared Actions and Session.undo; Apply calls the app owner.
    Canvas press captures width/mode; release commits one complete stroke,
    including an outside-axes release. Move is pointer-only preview.
    """

    def __init__(
        self,
        context: CrossSelectionContext,
        *,
        on_apply: Callable[[], CrossSelectionResult],
        submit_preview: CrossSelectionPreviewSubmitter | None = None,
    ) -> None:
        """Attach open context with Apply callback and 80ms-debounced projection.

        on_apply publishes via owner and retains input/Undo; failures appear in
        Status and remain editable. None submit_preview creates an owned runner.
        Injected submitter delivers on owner loop; caller owns external cleanup.
        Success/error must match current numerical snapshot/generation, attached
        widget and open input. Tool-only changes reproject without downsampling.
        Closed context raises FailedPreconditionError. Detach transfers no app
        ownership and does not cancel its session. Construct/use on the Qt owner.
        """
        context.session.ensure_input_open()
        super().__init__(controls_side="left")
        self._context = context
        self._on_apply = on_apply
        self._detached = False
        self._generation = 0
        self._view: CrossSelectionView | None = None
        self._observed = context.session.snapshot()
        self._previous: CrossSelectionState | None = None
        self._vertices: list[BrushPoint] = []
        self._gesture_width = 0.0
        self._gesture_mode: BrushMode = "select"
        self._plot = CrossSelectionPlot(self.figure, context.plugin.inputs)
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
        self._width = self._slider("Brush width")
        self._distance = self._slider("Min distance")
        self._mode = QtWidgets.QComboBox()
        self._mode.setAccessibleName("Operation")
        self._mode.addItems(["Select", "Erase"])
        self.controls_layout.addWidget(self._mode)
        self._width.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_tool.execute(
                    context.session, BrushTool(width=value / 1000)
                )
            )
        )
        self._distance.valueChanged.connect(
            lambda value: self._execute(
                lambda: context.plugin.set_min_distance.execute(
                    context.session, value / 1000
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
        self._button(
            "Perform on all",
            lambda: context.plugin.perform_on_all.execute(context.session, None),
        )
        self._button(
            "Clear", lambda: context.plugin.clear.execute(context.session, None)
        )
        self._undo = self._button("Undo", context.session.undo)
        self._button("Apply", self._apply_selection)
        self._show_changes = QtWidgets.QCheckBox("Show changes")
        self._show_changes.setAccessibleName("Show changes")
        self._show_changes.setChecked(True)
        self._show_changes.toggled.connect(self._render)
        self.controls_layout.addWidget(self._show_changes)
        self._status = QtWidgets.QLabel()
        self._status.setAccessibleName("Status")
        self._status.setWordWrap(True)
        self.controls_layout.addWidget(self._status)
        self.controls_layout.addStretch()
        self._unsubscribe = context.session.subscribe(self._show_committed)
        self._reproject_controls()
        self._schedule_preview()

    def preview_view(self) -> CrossSelectionView | None:
        """Read detached latest successful view, or None pending/failed/detached.

        Never calculate inline or return a stale generation or mutable cache.
        """
        return deepcopy(self._view) if self._input_open() else None

    def get_result(self) -> CrossSelectionResult:
        """Synchronously analyze the latest open snapshot, never cached preview.

        Does not publish, close input or consume Undo. Closed/detached view raises
        FailedPreconditionError; numerical invalid state raises ValueError.
        """
        if not self._input_open():
            raise FailedPreconditionError("cross-selection view is detached or closed")
        return analyze_cross_selection(
            self._context.plugin.inputs, self._context.session.snapshot()
        )

    def on_press(self, event: MouseEvent) -> None:
        """Begin a left-button gesture in plot axes, capturing committed tool."""
        if not self._input_open() or event.button != MouseButton.LEFT:
            return
        point = self._point(event)
        if point is None:
            return
        state = self._context.session.snapshot()
        self._gesture_width, self._gesture_mode = state.width, state.mode
        self._vertices = [point]
        self._render_pointer()

    def on_move(self, event: MouseEvent) -> None:
        """Append valid flux/GHz vertices to pointer preview, without commit."""
        if not self._input_open() or not self._vertices:
            return
        point = self._point(event)
        if point is not None and point != self._vertices[-1]:
            self._vertices.append(point)
            self._render_pointer()

    def on_release(self, event: MouseEvent) -> None:
        """Submit one stroke, using collected vertices even outside axes.

        Failed stroke makes no partial commit; restore controls and show error.
        """
        if not self._input_open() or not self._vertices:
            return
        point = self._point(event)
        if point is not None and point != self._vertices[-1]:
            self._vertices.append(point)
        stroke = BrushStroke(
            tuple(self._vertices), self._gesture_width, self._gesture_mode
        )
        self._vertices.clear()
        self._render_pointer()
        self._execute(
            lambda: self._context.plugin.stroke.execute(self._context.session, stroke)
        )

    def quiesce(self) -> None:
        """Disable this view, stop timer, join/flush owned runner on owner loop.

        Invalidate deliveries first. External cleanup remains caller's. Does not
        cancel the app-owned context. Safe to repeat; reattach with a new widget.
        """
        self._detached = True
        self._generation += 1
        self._timer.stop()
        self._view = None
        self._vertices.clear()
        self._render_pointer()
        if self._runner is not None:
            self._runner.quiesce()
        for control in self.findChildren(QtWidgets.QWidget):
            control.setEnabled(False)

    def teardown(self) -> None:
        """Permanently detach/unsubscribe and quiesce without cancelling context.

        Late success/error cannot update this view or Apply; safe to repeat.
        """
        self._unsubscribe()
        self.quiesce()

    def _slider(self, name: str) -> QtWidgets.QSlider:
        self.controls_layout.addWidget(QtWidgets.QLabel(name))
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setAccessibleName(name)
        slider.setRange(0, 100)
        self.controls_layout.addWidget(slider)
        return slider

    def _button(
        self, name: str, operation: Callable[[], object]
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(name)
        button.setAccessibleName(name)
        button.clicked.connect(lambda: self._execute(operation))
        self.controls_layout.addWidget(button)
        return button

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
            for control in self.findChildren(QtWidgets.QWidget):
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
            # Isolate Qt callbacks while preserving the committed state and failure.
            logger.exception("Cross-selection control failed")
            self._reproject_controls()
            self._status.setText(str(exc))

    def _apply_selection(self) -> None:
        result = self._on_apply()
        self._status.setText(
            f"Applied {np.count_nonzero(result.selected)}/{result.selected.size}; input remains open"
        )

    def _reproject_controls(self) -> None:
        state = self._context.session.snapshot()
        for slider, value in (
            (self._width, state.width),
            (self._distance, state.min_distance),
        ):
            blocked = slider.blockSignals(True)
            slider.setValue(round(value * 1000))
            slider.blockSignals(blocked)
        blocked = self._mode.blockSignals(True)
        self._mode.setCurrentText("Select" if state.mode == "select" else "Erase")
        self._mode.blockSignals(blocked)
        self._undo.setEnabled(self._context.session.can_undo())

    def _show_committed(self) -> None:
        if not self._input_open():
            return
        state = self._context.session.snapshot()
        changed = not _same_analysis(state, self._observed)
        if changed:
            # Analysis Actions create Undo; Undo consumes it before notifying.
            forward = self._context.session.can_undo()
            self._previous = None if forward else self._observed
        self._observed = state
        self._reproject_controls()
        if changed:
            self._schedule_preview()
        elif self._view is not None:
            self._view = replace(self._view, state=deepcopy(state))
            self._render()

    def _schedule_preview(self) -> None:
        self._generation += 1
        self._view = None
        self._status.setText("Updating selection…")
        self._timer.start()

    def _submit_owned(
        self,
        compute: Callable[[], CrossSelectionView],
        on_done: Callable[[CrossSelectionView], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._runner is None:
            raise RuntimeError("owned cross-selection runner is unavailable")
        self._runner.submit(compute, on_done=on_done, on_error=on_error)

    def _start_preview(self) -> None:
        if not self._input_open():
            return
        generation = self._generation
        state = self._context.session.snapshot()
        previous = deepcopy(self._previous)
        inputs = self._context.plugin.inputs

        def current() -> bool:
            return (
                generation == self._generation
                and self._input_open()
                and _same_analysis(state, self._context.session.snapshot())
            )

        def done(view: CrossSelectionView) -> None:
            if not current():
                return
            # Tool-only updates do not rerun downsampling; display their latest fields.
            self._view = replace(deepcopy(view), state=self._context.session.snapshot())
            self._render()
            self._status.setText(
                f"Selected {np.count_nonzero(view.result.selected)}/{view.result.selected.size}; "
                f"added {len(view.added_points)}, removed {len(view.removed_points)}"
            )

        def error(exc: Exception) -> None:
            if not current():
                return
            self._view = None
            self._status.setText(str(exc))

        try:
            self._submit_preview(
                lambda: project_cross_selection(inputs, state, previous=previous),
                done,
                error,
            )
        except Exception as exc:
            # Submission failure is presentation-only; Apply still computes synchronously.
            logger.exception("Cross-selection preview submission failed")
            error(exc)

    def _render(self) -> None:
        if self._input_open() and self._view is not None:
            self._plot.show_state(
                self._view, show_changes=self._show_changes.isChecked()
            )
            self.redraw()

    def _point(self, event: MouseEvent) -> BrushPoint | None:
        if (
            event.inaxes is not self.figure.axes[0]
            or event.xdata is None
            or event.ydata is None
            or not np.isfinite(event.xdata)
            or not np.isfinite(event.ydata)
        ):
            return None
        return BrushPoint(float(event.xdata), float(event.ydata))

    def _render_pointer(self) -> None:
        self._pointer.set_data(
            [p.x for p in self._vertices], [p.y for p in self._vertices]
        )
        self.redraw()


def _same_analysis(first: CrossSelectionState, second: CrossSelectionState) -> bool:
    if not (
        np.array_equal(first.selected, second.selected)
        and first.min_distance == second.min_distance
    ):
        return False
    a, b = first.last_change, second.last_change
    if a is None or b is None:
        return a is b
    return (
        np.array_equal(a.selected, b.selected)
        and a.min_distance == b.min_distance
        and a.vertices == b.vertices
        and a.width == b.width
    )
