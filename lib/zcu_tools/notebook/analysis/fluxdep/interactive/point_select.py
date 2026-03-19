from __future__ import annotations

from threading import Timer

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from typing_extensions import Optional

from zcu_tools.notebook.persistance import SpectrumResult
from zcu_tools.notebook.analysis.fluxdep.processing import (
    cast2real_and_norm,
    downsample_points,
)


class InteractiveSelector:
    def __init__(
        self,
        spectrums: dict[str, SpectrumResult],
        selected: Optional[NDArray[np.bool_]] = None,
        brush_width: float = 0.05,
    ) -> None:
        self.spectrums = spectrums

        self.s_fluxs = np.concatenate(
            [s["points"]["fluxs"] for s in spectrums.values()]
        )
        self.s_freqs = np.concatenate(
            [s["points"]["freqs"] for s in spectrums.values()]
        )

        self.selected = (
            selected if selected is not None else np.ones_like(self.s_fluxs, dtype=bool)
        )
        self.filter_mask = np.ones_like(self.selected, dtype=bool)

        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        plt.ion()

        self.set_plot_limit()
        self.create_widgets(brush_width)
        self.init_background(spectrums)
        self.init_points(self.s_fluxs, self.s_freqs, self.selected)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

        display(
            widgets.HBox(
                [
                    self.fig.canvas,
                    widgets.VBox(
                        [
                            self.width_slider,
                            self.thresh_slider,
                            self.operation_tb,
                            self.perform_all_bt,
                            self.finish_button,
                        ]
                    ),
                ]
            )
        )

        # 新增: 用於儲存暫時性圓圈和計時器的變數
        self.temp_circle = None
        self.temp_circle_timer = None

    def create_thresh_silders(self) -> None:
        self.thresh_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.1, step=1e-3, description="Min distance:"
        )

        self.thresh_slider.observe(
            lambda _: self.apply_filter_and_redraw(), names="value"
        )

    def create_widgets(self, brush_width) -> None:
        self.width_slider = widgets.FloatSlider(
            value=brush_width, min=0.01, max=0.1, step=1e-4, description="Brush Width:"
        )

        self.create_thresh_silders()

        self.operation_tb = widgets.Dropdown(
            options=["Select", "Erase"], value="Select", description="Operation:"
        )

        self.perform_all_bt = widgets.Button(
            description="Perform on All", button_style="warning"
        )
        self.perform_all_bt.on_click(self.on_perform_all)

        self.finish_button = widgets.Button(
            description="Finish", button_style="success"
        )
        self.finish_button.on_click(self.on_finish)

    def apply_filter_and_redraw(self) -> None:
        if self.is_finished:
            return

        sel_x = self.s_fluxs[self.selected] / (self.flux_bound[1] - self.flux_bound[0])
        sel_y = self.s_freqs[self.selected] / (self.freq_bound[1] - self.freq_bound[0])
        thresh = self.thresh_slider.value

        self.filter_mask = downsample_points(sel_x, sel_y, thresh)
        self.update_points()
        self.fig.canvas.draw_idle()

    def on_perform_all(self, _) -> None:
        if self.is_finished:
            return

        mode = self.operation_tb.value
        if mode == "Select":
            self.selected[:] = True
        elif mode == "Erase":
            self.selected[:] = False

        self.apply_filter_and_redraw()

    def init_background(self, s_pects: dict[str, SpectrumResult]) -> None:
        for spect in s_pects.values():
            # Get corresponding data and range
            signals = spect["spectrum"]["signals"] ** 1.5  # improve contrast
            flux_mask = np.any(~np.isnan(signals), axis=1)
            freq_mask = np.any(~np.isnan(signals), axis=0)
            signals = signals[flux_mask, :][:, freq_mask]

            # Normalize data
            real_signals = cast2real_and_norm(signals)

            # Add heatmap trace
            sp_fluxs = spect["spectrum"]["fluxs"][flux_mask]
            sp_freqs = spect["spectrum"]["freqs"][freq_mask]
            self.ax.imshow(
                real_signals.T,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(sp_fluxs[0], sp_fluxs[-1], sp_freqs[0], sp_freqs[-1]),
            )

    def init_points(self, s_fluxs, s_freqs, selected) -> None:
        self.scatter = self.ax.scatter(
            s_fluxs, s_freqs, c=selected.astype(float), s=2, vmax=1, vmin=0
        )

    def set_plot_limit(self) -> None:
        spect_list = [s["spectrum"] for s in self.spectrums.values()]
        sp_fluxs = np.concatenate([s["fluxs"] for s in spect_list])
        sp_freqs = np.concatenate([s["freqs"] for s in spect_list])
        self.flux_bound = (
            min(np.nanmin(sp_fluxs), self.s_fluxs.min()),
            max(np.nanmax(sp_fluxs), self.s_fluxs.max()),
        )
        self.freq_bound = (
            min(np.nanmin(sp_freqs), self.s_freqs.min()),
            max(np.nanmax(sp_freqs), self.s_freqs.max()),
        )

        # Set x and y axis range
        self.ax.set_xlim(self.flux_bound[0], self.flux_bound[1])
        self.ax.set_ylim(self.freq_bound[0], self.freq_bound[1])

    def get_cur_selected(self) -> NDArray[np.bool_]:
        cur_selected = np.zeros_like(self.selected, dtype=bool)
        cur_selected[np.where(self.selected)[0][self.filter_mask]] = True
        return cur_selected

    def update_points(self) -> None:
        self.scatter.set_array(self.get_cur_selected().astype(float))

    def toggle_near_mask(self, x, y, width):
        x_d = np.abs(self.s_fluxs - x) / (self.flux_bound[1] - self.flux_bound[0])
        y_d = np.abs(self.s_freqs - y) / (self.freq_bound[1] - self.freq_bound[0])
        toggle_mask = x_d**2 + y_d**2 <= width**2

        self.selected[toggle_mask] = self.operation_tb.value == "Select"

    def on_finish(self, _) -> None:
        self.finish_interactive()

        # also clear the output
        clear_output(wait=False)

    def finish_interactive(self) -> None:
        self.is_finished = True
        plt.close(self.fig)

    def get_positions(
        self, finish: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        if not self.is_finished and finish:
            self.finish_interactive()

        cur_selected = self.get_cur_selected()
        return (
            self.s_fluxs[cur_selected],
            self.s_freqs[cur_selected],
            cur_selected,
        )

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or self.is_finished:
            return

        # 計算靠近滑鼠點擊的點
        width = self.width_slider.value
        self.toggle_near_mask(event.xdata, event.ydata, width)

        # 新增: 顯示暫時性圓圈（並取消舊的計時器）
        self.show_temp_circle(event.xdata, event.ydata, width)
        self.apply_filter_and_redraw()

    def show_temp_circle(self, x, y, width) -> None:
        """顯示暫時性圓圈，一秒後消失"""
        # 移除現有的暫時性圓圈和計時器
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None
        if self.temp_circle_timer is not None:
            self.temp_circle_timer.cancel()
            self.temp_circle_timer = None

        # 計算圓圈的寬度和高度（考慮座標軸比例）
        x_range = self.flux_bound[1] - self.flux_bound[0]
        y_range = self.freq_bound[1] - self.freq_bound[0]

        # 根據當前模式決定顏色
        circle_color = "yellow" if self.operation_tb.value == "Select" else "black"

        # 使用 Ellipse 確保視覺上是正圓
        self.temp_circle = Ellipse(
            (x, y),
            width=width * x_range * 2,  # 直徑 = 半徑 * 2
            height=width * y_range * 2,
            angle=0,
            fill=False,
            color=circle_color,  # 修改這裡，根據模式選擇顏色
            linestyle="--",
            linewidth=1,
        )
        self.ax.add_patch(self.temp_circle)

        # 設置計時器一秒後移除圓圈
        self.temp_circle_timer = Timer(1.0, self.remove_temp_circle)
        self.temp_circle_timer.start()

    def remove_temp_circle(self) -> None:
        """移除暫時性圓圈"""
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None
            self.fig.canvas.draw_idle()
        self.temp_circle_timer = None
