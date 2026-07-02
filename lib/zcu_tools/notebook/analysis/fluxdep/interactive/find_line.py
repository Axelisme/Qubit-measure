from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep import TwoLinePicker


def _canvas_widget(canvas) -> widgets.Widget:
    if isinstance(canvas, widgets.Widget):
        return canvas
    output = widgets.Output()
    with output:
        display(canvas)
    return output


class InteractiveLines:
    TRACK_INFO = {
        "half": "<span style='color:red'>正在移動half flux(紅線)</span>",
        "integer": "<span style='color:blue'>正在移動integer flux(藍線)</span>",
        "none": "<span style='color:gray'>未選擇</span>",
    }

    def __init__(
        self,
        signals: NDArray,
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        flux_half: float | None = None,
        flux_int: float | None = None,
    ) -> None:
        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig = plt.figure(figsize=(5, 5))
        self.picker = TwoLinePicker(
            self.fig,
            np.asarray(signals, dtype=np.complex128),
            dev_values,
            freqs,
            flux_half=flux_half,
            flux_int=flux_int,
        )
        plt.ion()

        self.create_widgets()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)
        self.fig.canvas.mpl_connect("button_release_event", self.onrelease)

        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            self.flux_half_button,
                            self.flux_int_button,
                            self.auto_align_button,
                            self.swap_button,
                            self.finish_button,
                        ]
                    ),
                    widgets.HBox(
                        [
                            self.status_text,
                            self.position_text,
                            self.conjugate_checkbox,
                            self.only_use_magnitude_checkbox,
                        ]
                    ),
                    _canvas_widget(self.fig.canvas),
                ]
            )
        )

    @property
    def flux_half(self) -> float:
        return self.picker.positions()[0]

    @flux_half.setter
    def flux_half(self, value: float) -> None:
        _, flux_int = self.picker.positions()
        self.picker.apply_positions(value, flux_int)
        self._refresh_position()

    @property
    def flux_int(self) -> float:
        return self.picker.positions()[1]

    @flux_int.setter
    def flux_int(self, value: float) -> None:
        flux_half, _ = self.picker.positions()
        self.picker.apply_positions(flux_half, value)
        self._refresh_position()

    @property
    def only_use_magnitude(self) -> bool:
        return self.picker.magnitude_only

    def create_widgets(self) -> None:
        self.flux_half_button = widgets.Button(
            description="選擇half flux(紅線)",
            button_style="danger",
            tooltip="選擇紅色線進行移動",
        )
        self.flux_int_button = widgets.Button(
            description="選擇integer flux(藍線)",
            button_style="info",
            tooltip="選擇藍色線進行移動",
        )
        self.finish_button = widgets.Button(
            description="完成",
            button_style="success",
            tooltip="完成選擇並返回結果",
        )
        self.conjugate_checkbox = widgets.Checkbox(
            value=False, description="Conjugate Line"
        )
        self.only_use_magnitude_checkbox = widgets.Checkbox(
            value=self.picker.magnitude_only, description="Magnitude Only"
        )
        self.swap_button = widgets.Button(
            description="交換線條",
            button_style="warning",
            tooltip="交換half flux(紅線)與integer flux(藍線)的位置",
        )
        self.auto_align_button = widgets.Button(
            description="自動對齊",
            button_style="primary",
            tooltip="自動找到mirror loss最小的位置並移動",
        )
        self.position_text = widgets.HTML(value=self.get_info())
        self.status_text = widgets.HTML(value=self.TRACK_INFO["none"])

        self.flux_half_button.on_click(self.set_picked_half_flux)
        self.flux_int_button.on_click(self.set_picked_int_flux)
        self.finish_button.on_click(self.on_finish)
        self.swap_button.on_click(self.swap_lines)
        self.auto_align_button.on_click(self.auto_align_lines)
        self.conjugate_checkbox.observe(self.on_toggle_conjugate, names="value")
        self.only_use_magnitude_checkbox.observe(
            self.on_toggle_magnitude, names="value"
        )

    def get_info(self) -> str:
        flux_half, flux_int = self.picker.positions()
        period = 2 * abs(flux_int - flux_half)
        return (
            f"half flux: {flux_half:.2e}, "
            f"integer flux: {flux_int:.2e}, "
            f"flux period: {period:.2e}"
        )

    def _refresh_position(self) -> None:
        self.position_text.value = self.get_info()
        self.fig.canvas.draw_idle()

    def _refresh_status(self) -> None:
        role = self.picker.selected_role or "none"
        self.status_text.value = self.TRACK_INFO[role]

    def set_picked_half_flux(self, _) -> None:
        if self.is_finished:
            return
        if self.picker.selected_role == "half":
            self.picker.clear_selection()
        else:
            self.picker.pick_half()
        self._refresh_status()

    def set_picked_int_flux(self, _) -> None:
        if self.is_finished:
            return
        if self.picker.selected_role == "integer":
            self.picker.clear_selection()
        else:
            self.picker.pick_integer()
        self._refresh_status()

    def stop_tracking(self) -> None:
        self.picker.clear_selection()
        self._refresh_status()

    def swap_lines(self, _) -> None:
        if self.is_finished:
            return
        self.picker.swap()
        self._refresh_status()
        self._refresh_position()

    def auto_align_lines(self, _) -> None:
        if self.is_finished:
            return
        self.picker.auto_align()
        self._refresh_status()
        self._refresh_position()

    def onclick(self, event) -> None:
        if self.is_finished or not self.picker.is_main_axes(event.inaxes):
            return
        self.picker.on_press(event.xdata)
        self._refresh_status()

    def onmove(self, event) -> None:
        if self.is_finished or not self.picker.is_main_axes(event.inaxes):
            return
        self.picker.on_move(event.xdata)
        self._refresh_position()

    def onrelease(self, event) -> None:
        if self.is_finished or not self.picker.is_main_axes(event.inaxes):
            return
        self.picker.on_release(event.xdata, event.ydata)

    def on_finish(self, _) -> None:
        self.finish_interactive()
        clear_output(wait=False)

    def finish_interactive(self) -> None:
        self.is_finished = True
        self.picker.clear_selection()
        self._refresh_status()
        plt.close(self.fig)

    def get_positions(self, finish: bool = True) -> tuple[float, float]:
        if not self.is_finished and finish:
            self.finish_interactive()
        return self.picker.positions()

    def on_toggle_conjugate(self, change) -> None:
        if self.is_finished:
            return
        self.picker.set_conjugate(bool(change["new"]))

    def on_toggle_magnitude(self, change) -> None:
        if self.is_finished:
            return
        self.picker.set_magnitude_only(bool(change["new"]))
        self._refresh_status()
        self._refresh_position()
