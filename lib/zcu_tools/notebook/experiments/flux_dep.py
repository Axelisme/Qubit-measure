"""Notebook control for OneTone flux-dependent sweeps and committed picks."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython import display as ipython_display
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    fold_initial_lines,
)
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepAnalysis,
    FluxDepAnalyzeOptions,
    FluxDepCfg,
    FluxDepResult,
)
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepExp as FluxDepCore,
)
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost, Plots


@dataclass(frozen=True)
class FluxDepAnalysisRecord:
    source: FluxDepResult
    options: FluxDepAnalyzeOptions
    result: FluxDepAnalysis
    plots: Plots


class FluxDepInteraction:
    """Preview a selection; only an explicit Done publishes its core analysis."""

    def __init__(  # noqa: PLR0913 - Notebook interaction inputs are explicit
        self,
        source: FluxDepResult,
        core: FluxDepCore,
        host: PlotHost,
        publish: Callable[[FluxDepAnalysisRecord], None],
        *,
        flux_half: float | None,
        flux_int: float | None,
        conjugate: bool,
        magnitude_only: bool,
        present: bool,
    ) -> None:
        inputs = FluxPickInputs(source.signals, source.values, source.freqs)
        half, integer = fold_initial_lines(inputs.dev_values, flux_half, flux_int)
        seed = FluxPickState(
            flux_half=half,
            flux_int=integer,
            conjugate=conjugate,
            magnitude_only=magnitude_only,
        )
        self._source = source
        self._core = core
        self._host = host
        self._publish = publish
        self._present = present
        self.is_finished = False
        self.result: FluxDepAnalysis | None = None
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
        if present:
            try:
                ipython_display.display(self.widget)
                host.present(self.figure)
            except BaseException:
                try:
                    host.release(self.figure)
                finally:
                    self._close_controls()
                raise
        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)

    def positions(self) -> tuple[float, float]:
        """Read current positions without auto-completing the interaction."""
        return self.picker.positions()

    def set_positions(self, half: float, integer: float) -> None:
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        self.picker.apply_positions(half, integer)
        self._refresh()

    def done(self) -> FluxDepAnalysis:
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        half, integer = self.picker.positions()
        options = FluxDepAnalyzeOptions(
            half,
            integer,
            conjugate=self.conjugate_checkbox.value,
            magnitude_only=self.magnitude_checkbox.value,
        )
        plots = Plots(self._host)
        try:
            result = self._core.analyze(self._source, options, plots=plots)
            plots.finish()
        except BaseException:
            try:
                plots.finish(present=False)
            finally:
                plots.release()
            raise
        self._host.release(self.figure)
        self.picker.clear_selection()
        self._close_controls()
        record = FluxDepAnalysisRecord(self._source, options, result, plots)
        self._publish(record)
        self.result = result
        self.plots = plots
        self.is_finished = True
        return result

    def cancel(self) -> None:
        if self.is_finished:
            raise RuntimeError("Flux interaction has finished")
        self._host.release(self.figure)
        self.picker.clear_selection()
        self._close_controls()
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


class FluxDepNotebookExp:
    """Notebook convenience over stateless OneTone acquisition and analysis."""

    def __init__(self, *, present: bool = True) -> None:
        self._core = FluxDepCore()
        self._host: PlotHost = NotebookPlotHost() if present else NonPresentingHost()
        self._present = present
        self.last_result: FluxDepResult | None = None
        self.run_plots: Plots | None = None
        self._analysis: FluxDepAnalysisRecord | None = None

    @property
    def analysis(self) -> FluxDepAnalysisRecord | None:
        return self._analysis

    @property
    def analysis_plots(self) -> Plots:
        if self._analysis is None:
            raise RuntimeError("No successful FluxDep analysis")
        return self._analysis.plots

    def run(self, soc: Any, soccfg: Any, cfg: FluxDepCfg) -> FluxDepResult:
        plots = Plots(self._host)
        try:
            result = self._core.run(cfg, context=QickContext(soc, soccfg, plots))
            plots.finish()
        except BaseException:
            try:
                plots.finish(present=False)
            finally:
                plots.release()
            raise
        self.last_result = result
        self.run_plots = plots
        self._analysis = None
        return result

    def analyze(
        self,
        result: FluxDepResult | None = None,
        *,
        flux_half: float | None = None,
        flux_int: float | None = None,
        conjugate: bool = False,
        magnitude_only: bool = False,
    ) -> FluxDepInteraction:
        source = self.last_result if result is None else result
        if source is None:
            raise ValueError("No FluxDep result to analyze")
        return FluxDepInteraction(
            source,
            self._core,
            self._host,
            self._publish,
            flux_half=flux_half,
            flux_int=flux_int,
            conjugate=conjugate,
            magnitude_only=magnitude_only,
            present=self._present,
        )

    def save(
        self,
        filepath: str | Path,
        result: FluxDepResult | None = None,
        comment: str | None = None,
        tag: str | None = None,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        selected = self.last_result if result is None else result
        if selected is None:
            raise ValueError("No FluxDep result to save")
        self._core.save(
            selected,
            Path(filepath),
            comment=comment,
            tag=tag,
            server_ip=server_ip,
            port=port,
        )

    def load(
        self,
        filepath: str | Path,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> FluxDepResult:
        result = self._core.load(Path(filepath), server_ip=server_ip, port=port)
        self.last_result = result
        self.run_plots = None
        self._analysis = None
        return result

    def _publish(self, record: FluxDepAnalysisRecord) -> None:
        self._analysis = record
