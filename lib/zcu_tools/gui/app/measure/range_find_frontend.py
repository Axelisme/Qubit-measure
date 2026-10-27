"""Qt spectrum picker; numeric edits commit through the captured plugin session."""

from collections.abc import Callable

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qtpy.QtGui import QResizeEvent
from qtpy.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.analysis.dips import DipFinderOptions
from zcu_tools.gui.app.measure.range_find_plugin import RangeFindPlugin, RangeFindState
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.plotting.plots import Plots


def draw_range_find(
    figure: Figure,
    freqs: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    state: RangeFindState,
) -> None:
    """Replace figure with measured magnitude, candidates and accepted selections.

    freqs/magnitude are the plugin's sorted MHz/magnitude arrays. state supplies
    current candidates and manual selections. Manual points use measured
    magnitude; no fitted values are invented. Runs on the figure owner.
    """
    figure.clear()
    ax = figure.subplots()
    ax.plot(freqs, magnitude, label="Measured magnitude", linewidth=1)
    if state.candidates:
        x = [dip.freq for dip in state.candidates]
        ax.plot(x, np.interp(x, freqs, magnitude), "o", color="0.6", label="Candidates")
    for dip in state.selected:
        y = float(np.interp(dip.freq, freqs, magnitude))
        ax.plot(
            dip.freq,
            y,
            "*",
            color="crimson",
            markersize=12,
            label=f"{dip.component} @ {dip.freq:g} MHz"
            if dip.component is not None
            else None,
        )
        if dip.component is not None:
            ax.annotate(
                dip.component, (dip.freq, y), xytext=(6, 10), textcoords="offset points"
            )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.legend(loc="best", fontsize="small")
    figure.set_layout_engine("constrained")


class RangeFindFrontend[R](InteractiveFrontend):
    """Render and edit one borrowed RangeFindPlugin session on the Qt owner thread.

    Click near a selected star to remove it; other spectrum clicks add the nearest
    sample. Controls recompute on Enter/focus loss, replacing manual selections
    while retaining assignments at surviving frequencies. The selected-dip and
    component boxes assign caller labels without frequency-order restrictions.
    request_finish/request_cancel belong to the host; teardown retires all
    callbacks but never disposes its borrowed session. No pointer preview exists.
    """

    def __init__(
        self,
        plugin: RangeFindPlugin[R],
        session: Session[RangeFindState],
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plugin, self._session = plugin, session
        self._request_finish, self._request_cancel = request_finish, request_cancel
        self._read_only = False
        self._figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setObjectName("rangeFind.spectrum")
        self._fields: dict[str, QLineEdit] = {}
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        plot = QWidget(self)
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas, stretch=1)
        self._controls = self._make_controls()
        self._side_controls_width = self._controls.minimumSizeHint().width()
        self._root = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._root.addWidget(plot, stretch=1)
        self._root.addWidget(self._controls)
        connection = self._canvas.mpl_connect(
            "button_press_event", self._scope.callback(self._click)
        )
        self._scope.add_cleanup(lambda: self._canvas.mpl_disconnect(connection))
        self._scope.add_cleanup(session.subscribe(self._scope.callback(self._refresh)))
        self._scope.add_cleanup(lambda: self.set_read_only(read_only=True))
        self._refresh()

    def _make_controls(self) -> QWidget:
        controls = QWidget(self)
        controls.setMinimumWidth(280)
        self._controls_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, controls)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        self._controls_layout.addLayout(column)
        actions = QVBoxLayout()
        self._controls_layout.addLayout(actions)
        form = QFormLayout()
        for name, label in (
            ("smoothing_sigma", "Smoothing sigma (samples)"),
            ("prominence_factor", "Prominence noise factor"),
            ("max_fwhm", "Max FWHM (MHz)"),
            ("min_separation", "Min separation (MHz)"),
            ("max_dips", "Resonator count (0 = all)"),
        ):
            field = QLineEdit()
            field.setObjectName(f"rangeFind.{name}")
            self._scope.connect(field.editingFinished, self._parameters_changed)
            self._fields[name] = field
            form.addRow(label, field)
        self._selection = QComboBox()
        self._selection.setObjectName("rangeFind.selection")
        self._component = QComboBox()
        self._component.setObjectName("rangeFind.component")
        self._component.addItem("Unassigned")
        self._component.addItems(list(self._plugin.components))
        self._component.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._component.setMinimumWidth(self._component.sizeHint().width())
        form.addRow("Selected dip", self._selection)
        form.addRow("Component", self._component)
        column.addLayout(form)
        self._scope.connect(self._selection.currentIndexChanged, self._show_assignment)
        self._scope.connect(self._component.activated, self._assign)
        self._remove = QPushButton("Remove selected dip")
        self._scope.connect(self._remove.clicked, self._remove_selected)
        actions.addWidget(self._remove)
        hint = QLabel(
            "Click a star to remove it.\nClick elsewhere on the spectrum to add a sample.\nParameter edits replace manual selections."
        )
        hint.setWordWrap(True)
        actions.addWidget(hint)
        self._status = QLabel()
        self._status.setObjectName("rangeFind.status")
        self._status.setWordWrap(True)
        actions.addWidget(self._status)
        actions.addStretch()
        self._done, self._cancel = QPushButton("Done"), QPushButton("Cancel")
        self._scope.connect(self._done.clicked, self._finish)
        self._scope.connect(self._cancel.clicked, self._cancel_analysis)
        actions.addWidget(self._done)
        actions.addWidget(self._cancel)
        return controls

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """Keep controls below the plot when a side column would leave <500 px."""
        super().resizeEvent(a0)
        margins = self._root.contentsMargins()
        available = self.width() - margins.left() - margins.right()
        below = available - self._side_controls_width - self._root.spacing() < 500
        self._root.setDirection(
            QBoxLayout.Direction.TopToBottom
            if below
            else QBoxLayout.Direction.LeftToRight
        )
        self._controls_layout.setDirection(
            QBoxLayout.Direction.LeftToRight
            if below
            else QBoxLayout.Direction.TopToBottom
        )

    @property
    def figure(self) -> Figure:
        """Borrow the currently rendered spectrum; final plots are separately owned."""
        return self._figure

    @property
    def preview_active(self) -> bool:
        """False: every accepted edit immediately commits to the session."""
        return False

    def cancel_preview(self) -> None:
        """No pointer preview is retained by this frontend."""

    def set_read_only(self, *, read_only: bool) -> None:
        """Disable all editing while preserving the spectrum for viewing."""
        self._read_only = read_only
        for widget in (
            *self._fields.values(),
            self._selection,
            self._component,
            self._remove,
            self._done,
            self._cancel,
        ):
            widget.setEnabled(not read_only and not self._scope.closed)

    def _refresh(self) -> None:
        state = self._session.snapshot()
        values = {
            "smoothing_sigma": state.options.smoothing_sigma,
            "prominence_factor": state.options.prominence_factor,
            "max_fwhm": state.options.max_fwhm,
            "min_separation": state.options.min_separation,
            "max_dips": state.options.max_dips or 0,
        }
        for name, value in values.items():
            self._fields[name].setText(str(value))
        previous = self._selection.currentIndex()
        self._selection.blockSignals(True)
        self._selection.clear()
        self._selection.addItems([f"{dip.freq:g} MHz" for dip in state.selected])
        self._selection.setCurrentIndex(min(max(previous, 0), len(state.selected) - 1))
        self._selection.blockSignals(False)
        self._show_assignment()
        self._status.setText(
            f"Selected: {len(state.selected)} / candidates: {len(state.candidates)}"
        )
        draw_range_find(self._figure, self._plugin.freqs, self._plugin.magnitude, state)
        self._canvas.draw_idle()

    def _show_assignment(self) -> None:
        index = self._selection.currentIndex()
        selected = self._session.snapshot().selected
        component = selected[index].component if 0 <= index < len(selected) else None
        self._component.setCurrentIndex(
            0 if component is None else self._plugin.components.index(component) + 1
        )

    def _parameters_changed(self) -> bool:
        if self._read_only:
            return False
        try:
            count = int(self._fields["max_dips"].text())
            options = DipFinderOptions(
                smoothing_sigma=float(self._fields["smoothing_sigma"].text()),
                prominence_factor=float(self._fields["prominence_factor"].text()),
                max_fwhm=float(self._fields["max_fwhm"].text()),
                min_separation=float(self._fields["min_separation"].text()),
                max_dips=None if count == 0 else count,
            )
            if options != self._session.snapshot().options:
                self._plugin.set_parameters.execute(self._session, options)
        except ValueError as exc:
            self._status.setText(str(exc))
            return False
        return True

    def _assign(self, index: int) -> None:
        if not self._read_only and self._selection.currentIndex() >= 0:
            component = None if index == 0 else self._plugin.components[index - 1]
            self._plugin.assign_dip.execute(
                self._session, (self._selection.currentIndex(), component)
            )

    def _remove_selected(self) -> None:
        index = self._selection.currentIndex()
        if not self._read_only and index >= 0:
            self._plugin.remove_dip.execute(self._session, index)

    def _click(self, event: MouseEvent) -> None:
        if (
            self._read_only
            or self._toolbar.mode
            or event.button != MouseButton.LEFT
            or event.xdata is None
            or event.inaxes not in self._figure.axes
        ):
            return
        state = self._session.snapshot()
        ax = self._figure.axes[0]
        for index, dip in enumerate(state.selected):
            y = float(np.interp(dip.freq, self._plugin.freqs, self._plugin.magnitude))
            xpx, ypx = ax.transData.transform((dip.freq, y))
            if np.hypot(event.x - xpx, event.y - ypx) <= 12:
                self._plugin.remove_dip.execute(self._session, index)
                return
        try:
            self._plugin.add_dip.execute(self._session, event.xdata)
        except InvalidInputError as exc:
            self._status.setText(str(exc))

    def _finish(self) -> None:
        if not self._parameters_changed():
            return
        try:
            if self._request_finish():
                self.teardown()
        except FailedPreconditionError as exc:
            self._status.setText(str(exc))

    def _cancel_analysis(self) -> None:
        if not self._read_only and self._request_cancel():
            self.teardown()


def make_range_find_frontend[R](
    plugin: PluginDefinition[RangeFindState, R],
    session: Session[RangeFindState],
    env: InteractiveFrontendEnv,
    request_finish: Callable[[], bool],
    request_cancel: Callable[[], bool],
    *,
    plots: Plots,
) -> RangeFindFrontend[R]:
    """Mount a RangeFindPlugin/session pair; host retains session/plot ownership.

    env and plots are host protocol arguments; this synchronous frontend submits
    no workers. A non-RangeFindPlugin raises TypeError before widget allocation.
    """
    del env, plots
    if not isinstance(plugin, RangeFindPlugin):
        raise TypeError("range find frontend requires RangeFindPlugin")
    return RangeFindFrontend(plugin, session, request_finish, request_cancel)
