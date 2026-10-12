"""Notebook control for flux-dependent spectra and committed picks."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

import ipywidgets as widgets
from IPython import display as ipython_display
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from typing_extensions import TypeVar

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
    fold_initial_lines,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.plots import PlotHost, Plots


@dataclass(frozen=True)
class FluxDepPickerOptions:
    """Initial device-axis positions and display settings, not terminal options.

    flux_half and flux_int are finite device-axis positions; None selects the
    line kernel's initial defaults. conjugate couples movement of both lines.
    magnitude_only projects complex signals to magnitude for display/alignment.
    These settings seed the preview; Done records its actual terminal state."""

    flux_half: float | None = None
    flux_int: float | None = None
    conjugate: bool = False
    magnitude_only: bool = False


CfgT = TypeVar("CfgT", bound=ExpCfgModel, default=ExpCfgModel)
ResultT = TypeVar("ResultT", default=object)

FluxDepAnalysisRecord = AnalysisRecord[CfgT, ResultT, FluxPickState, FluxPickAnalysis]


class FluxDepInteraction(Generic[CfgT, ResultT]):
    """Preview caller-supplied inputs and publish only after successful Done.

    source is an opaque RunRecord retained by identity, never read for spectra.
    inputs contains the independent complex spectrum and device/frequency axes;
    options seeds positions/display. host owns figure presentation; publish
    receives the new AnalysisRecord and Plots only after cleanup succeeds.
    Construction displays widgets and preview; presentation errors propagate
    with cleanup causes. Use on the Notebook owner thread.

    figure/picker are the disposable preview, widget and the named buttons,
    checkboxes/status are live controls. is_finished marks retired controls.
    record and plots are None until Done succeeds, then hold the committed
    native figure and its presentation handle. Caller releases successful Plots;
    Cancel retires only this preview and never changes a previous analysis."""

    def __init__(
        self,
        source: RunRecord[CfgT, ResultT],
        inputs: FluxPickInputs,
        options: FluxDepPickerOptions,
        host: PlotHost,
        publish: Callable[[FluxDepAnalysisRecord[CfgT, ResultT], Plots], None],
    ) -> None:
        half, integer = fold_initial_lines(
            inputs.dev_values, options.flux_half, options.flux_int
        )
        conjugate = options.conjugate
        magnitude_only = options.magnitude_only
        seed = FluxPickState(
            flux_half=half,
            flux_int=integer,
            conjugate=conjugate,
            magnitude_only=magnitude_only,
        )
        self._source = source
        self._inputs = inputs
        self._host = host
        self._publish = publish
        self.is_finished = False
        self.record: FluxDepAnalysisRecord[CfgT, ResultT] | None = None
        self.plots: Plots | None = None
        self.figure = Figure(figsize=(8, 5))
        FigureCanvasAgg(self.figure)
        self.picker = TwoLinePicker(
            self.figure,
            inputs.signals,
            inputs.dev_values,
            inputs.freqs,
            flux_half=half,
            flux_int=integer,
            force_magnitude=magnitude_only,
        )
        self.picker.show_state(seed)

        self.half_button = widgets.Button(description="Select half")
        self.integer_button = widgets.Button(description="Select integer")
        self.swap_button = widgets.Button(description="Swap")
        self.align_button = widgets.Button(description="Align")
        self.done_button = widgets.Button(description="Done", button_style="success")
        self.cancel_button = widgets.Button(description="Cancel")
        self.conjugate_checkbox = widgets.Checkbox(
            value=conjugate, description="Conjugate"
        )
        self.magnitude_checkbox = widgets.Checkbox(
            value=magnitude_only, description="Magnitude only"
        )
        self.status = widgets.HTML(value=self.picker.info_text())
        self._button_row = widgets.HBox(
            [
                self.half_button,
                self.integer_button,
                self.swap_button,
                self.align_button,
                self.done_button,
                self.cancel_button,
            ]
        )
        self._option_row = widgets.HBox(
            [self.conjugate_checkbox, self.magnitude_checkbox]
        )
        self.widget = widgets.VBox([self._button_row, self._option_row, self.status])
        self.half_button.on_click(lambda _button: self._select("half"))
        self.integer_button.on_click(lambda _button: self._select("integer"))
        self.swap_button.on_click(lambda _button: self._change(self.picker.swap))
        self.align_button.on_click(lambda _button: self._change(self.picker.auto_align))
        self.done_button.on_click(self._on_done)
        self.cancel_button.on_click(lambda _button: self.cancel())
        self.conjugate_checkbox.observe(self._set_conjugate, names="value")
        self.magnitude_checkbox.observe(self._set_magnitude, names="value")
        try:
            ipython_display.display(self.widget)
            host.present(self.figure)
        except BaseException as error:
            try:
                self._retire_preview()
            except BaseException as cleanup_error:  # noqa: BLE001 - retain both operation failures
                raise BaseExceptionGroup(
                    "Flux interaction start and cleanup failed", [error, cleanup_error]
                ) from None
            raise
        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)

    def positions(self) -> tuple[float, float]:
        """Return current (half, integer) device-axis positions without finishing."""
        return self.picker.positions()

    def set_positions(self, half: float, integer: float) -> None:
        """Set preview half/integer in device-axis units and refresh the display.

        Non-finite positions raise ValueError; a finished interaction raises
        RuntimeError. This updates preview only, not the analyzer's saved record."""
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        self.picker.apply_positions(half, integer)
        self._refresh()

    def done(self) -> FluxDepAnalysisRecord[CfgT, ResultT]:
        """Commit the captured source, terminal state and native pick figure.

        Invalid separation leaves this interaction editable. Other failures retire
        this operation without replacing a previous successful record or plots.
        """
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        half, integer = self.picker.positions()
        state = FluxPickState(
            flux_half=half,
            flux_int=integer,
            conjugate=self.conjugate_checkbox.value,
            magnitude_only=self.magnitude_checkbox.value,
        )
        try:
            result = analyze_flux_pick(self._inputs, state)
        except ValueError:
            raise
        except BaseException as error:
            try:
                self._retire_preview()
            except BaseException as cleanup_error:  # noqa: BLE001 - retain both operation failures
                raise BaseExceptionGroup(
                    "Flux analysis and preview cleanup failed", [error, cleanup_error]
                ) from None
            raise
        plots = Plots(self._host)
        try:
            plots.adopt("pick", make_flux_pick_figure(self._inputs, state))
            figures = plots.finish()
            record = FluxDepAnalysisRecord[CfgT, ResultT](
                self._source, state, result, figures
            )
            self._retire_preview()
        except BaseException as error:
            cleanup_errors: list[BaseException] = []
            for cleanup in (lambda: plots.finish(present=False), plots.release):
                try:
                    cleanup()
                except BaseException as cleanup_error:  # noqa: BLE001 - continue operation cleanup
                    cleanup_errors.append(cleanup_error)
            if not self.is_finished:
                try:
                    self._retire_preview()
                except BaseException as cleanup_error:  # noqa: BLE001 - retain the preview failure
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Flux analysis and cleanup failed", [error, *cleanup_errors]
                ) from None
            raise
        self._publish(record, plots)
        self.record = record
        self.plots = plots
        return record

    def cancel(self) -> None:
        """Retire this preview and close controls without publishing a record.

        Finished interactions raise RuntimeError. Presentation cleanup errors
        propagate; controls still close and is_finished becomes True."""
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        self._retire_preview()

    def _retire_preview(self) -> None:
        try:
            self._host.release(self.figure)
        finally:
            self.picker.clear_selection()
            try:
                self._close_controls()
            finally:
                self.is_finished = True

    def _close_controls(self) -> None:
        for control in (
            self.half_button,
            self.integer_button,
            self.swap_button,
            self.align_button,
            self.done_button,
            self.cancel_button,
            self.conjugate_checkbox,
            self.magnitude_checkbox,
            self.status,
        ):
            control.style.close()
            control.layout.close()
            control.close()
        for row in (self._button_row, self._option_row):
            row.layout.close()
            row.close()
        self.widget.layout.close()
        self.widget.close()

    def _refresh(self) -> None:
        self.status.value = self.picker.info_text()
        self.figure.canvas.draw_idle()

    def _change(self, action: Callable[[], None]) -> None:
        if not self.is_finished:
            action()
            self._refresh()

    def _select(self, role: str) -> None:
        if self.is_finished:
            return
        if self.picker.selected_role == role:
            self.picker.clear_selection()
        elif role == "half":
            self.picker.pick_half()
        else:
            self.picker.pick_integer()
        self._refresh()

    def _set_conjugate(self, change: dict[str, Any]) -> None:
        if not self.is_finished:
            self.picker.set_conjugate(bool(change["new"]))
            self._refresh()

    def _set_magnitude(self, change: dict[str, Any]) -> None:
        if not self.is_finished:
            self.picker.set_magnitude_only(bool(change["new"]))
            self._refresh()

    def _on_press(self, event: MouseEvent) -> None:
        if not self.is_finished and self.picker.is_main_axes(event.inaxes):
            self.picker.on_press(event.xdata)

    def _on_move(self, event: MouseEvent) -> None:
        if not self.is_finished and self.picker.is_main_axes(event.inaxes):
            self.picker.on_move(event.xdata)
            self._refresh()

    def _on_release(self, event: MouseEvent) -> None:
        if not self.is_finished and self.picker.is_main_axes(event.inaxes):
            self.picker.on_release(event.xdata, event.ydata)
            self._refresh()

    def _on_done(self, _button: widgets.Button) -> None:
        try:
            self.done()
        except ValueError as error:
            self.status.value = str(error)


class FluxDepAnalyzer(Generic[CfgT, ResultT]):
    """Analyze explicit records independently of acquisition and persistence.

    host is the caller's owner-thread PlotHost; None creates NotebookPlotHost.
    analysis and analysis_plots start as None and hold the latest successful
    AnalysisRecord and presentation handle. Failed/cancelled previews do not
    replace them. The caller releases retained Plots; run/load elsewhere never
    changes this tool. No concrete experiment cfg/result is interpreted."""

    def __init__(self, host: PlotHost | None = None) -> None:
        self._host = NotebookPlotHost() if host is None else host
        self.analysis: FluxDepAnalysisRecord[CfgT, ResultT] | None = None
        self.analysis_plots: Plots | None = None

    def start(
        self,
        source: RunRecord[CfgT, ResultT],
        inputs: FluxPickInputs,
        options: FluxDepPickerOptions | None = None,
    ) -> FluxDepInteraction[CfgT, ResultT]:
        """Display a preview for opaque source using the caller's separate inputs.

        options supplies initial positions/display or None for kernel defaults.
        Returns an editable FluxDepInteraction. Spectral/position or presentation
        failures propagate without replacing analysis/analysis_plots. The caller
        finishes or cancels each preview; successful Done publishes this source by
        identity, terminal state, numeric pick result and a named native pick figure."""
        return FluxDepInteraction(
            source,
            inputs,
            FluxDepPickerOptions() if options is None else options,
            self._host,
            self._publish,
        )

    def _publish(
        self, record: FluxDepAnalysisRecord[CfgT, ResultT], plots: Plots
    ) -> None:
        self.analysis = record
        self.analysis_plots = plots
