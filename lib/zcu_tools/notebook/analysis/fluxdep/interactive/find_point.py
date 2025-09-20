from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from numpy import ndarray

from ..processing import cast2real_and_norm, spectrum2d_findpoint


class InteractiveFindPoints:
    def __init__(self, spectrum, mAs, fpts, threshold=1.0, brush_width=0.05) -> None:
        self.spectrum = spectrum
        self.mAs = mAs
        self.fpts = fpts

        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        plt.ion()

        # 顯示 widget
        self.create_widgets(threshold, brush_width)

        # 顯示頻譜
        self.init_background(spectrum, mAs, fpts)

        # 顯示 mask
        self.init_mask(fpts, mAs)

        # 顯示發現的點
        self.init_points(mAs, fpts, spectrum)

        # 準備手繪曲線
        self.init_callback()

        display(
            widgets.HBox(
                [
                    self.fig.canvas,
                    widgets.VBox(
                        [
                            self.threshold_slider,
                            self.width_slider,
                            self.smooth_slider,
                            self.operation_tb,
                            widgets.HBox(
                                [
                                    self.show_mask_box,
                                    self.show_origin_box,
                                    widgets.VBox(
                                        [self.perform_all_bt, self.finish_button]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            )
        )

    def create_widgets(self, threshold, brush_width) -> None:
        self.threshold_slider = widgets.FloatSlider(
            value=threshold, min=1.0, max=20.0, step=0.01, description="Threshold:"
        )
        self.width_slider = widgets.FloatSlider(
            value=brush_width, min=0.01, max=0.1, step=1e-4, description="Brush Width:"
        )
        self.smooth_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=5.0, step=0.01, description="Smooth:"
        )
        self.show_mask_box = widgets.Checkbox(value=False, description="Show Mask")
        self.show_origin_box = widgets.Checkbox(value=True, description="Show Origin")
        self.operation_tb = widgets.Dropdown(
            options=["Select", "Erase"], value="Select", description="Operation:"
        )
        self.perform_all_bt = widgets.Button(
            description="Perform on All", button_style="danger"
        )
        self.finish_button = widgets.Button(
            description="Finish", button_style="success"
        )

        self.threshold_slider.observe(self.on_ratio_change, names="value")
        self.smooth_slider.observe(self.on_ratio_change, names="value")
        self.show_mask_box.observe(self.on_select_show, names="value")
        self.show_origin_box.observe(self.on_show_origin_change, names="value")
        self.perform_all_bt.on_click(self.on_perform_all)
        self.finish_button.on_click(self.on_finish)

    def init_background(self, spectrum, mAs, fpts) -> None:
        amps = cast2real_and_norm(spectrum, sigma=self.smooth_slider.value)

        dx = (mAs[-1] - mAs[0]) / (len(mAs) - 1)
        dy = (fpts[-1] - fpts[0]) / (len(fpts) - 1)
        self.spectrum_img = self.ax.imshow(
            amps,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(
                mAs[0] - dx / 2,
                mAs[-1] + dx / 2,
                fpts[0] - dy / 2,
                fpts[-1] + dy / 2,
            ),
        )

    def init_mask(self, fpts, mAs) -> None:
        self.mask = np.ones((len(fpts), len(mAs)), dtype=bool)

        self.select_mask = self.ax.imshow(
            self.mask,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(mAs[0], mAs[-1], fpts[0], fpts[-1]),
            alpha=0.2 if self.show_mask_box.value else 0,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

    def init_points(self, mAs, fpts, spectrum) -> None:
        threshold = self.threshold_slider.value

        amps = cast2real_and_norm(spectrum, sigma=self.smooth_slider.value)
        self.s_mAs, self.s_fpts = spectrum2d_findpoint(
            mAs, fpts, amps, threshold, weight=self.mask
        )
        self.scatter = self.ax.scatter(self.s_mAs, self.s_fpts, color="r", s=2)

    def init_callback(self) -> None:
        # 綁定事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

    def update_points(self) -> None:
        threshold = self.threshold_slider.value

        amps = cast2real_and_norm(self.spectrum, sigma=self.smooth_slider.value)
        self.s_mAs, self.s_fpts = spectrum2d_findpoint(
            self.mAs, self.fpts, amps, threshold, weight=self.mask
        )
        self.scatter.set_offsets(np.column_stack((self.s_mAs, self.s_fpts)))

        # Set spectrum image data based on show_origin checkbox
        if self.show_origin_box.value:
            self.spectrum_img.set_data(amps)
        else:
            self.spectrum_img.set_data(self.mask * amps)
        self.spectrum_img.autoscale()

    def toggle_near_mask(self, x, y, width, mask, mode) -> None:
        x_d = np.abs(self.mAs - x) / (self.mAs[-1] - self.mAs[0])
        y_d = np.abs(self.fpts - y) / (self.fpts[-1] - self.fpts[0])
        d2 = x_d[None, :] ** 2 + y_d[:, None] ** 2

        weight = d2 <= width**2
        if mode == "Select":
            mask |= weight
        elif mode == "Erase":
            mask &= ~weight

    def on_ratio_change(self, _) -> None:
        if self.is_finished:
            return

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_select_show(self, _) -> None:
        if self.is_finished:
            return

        if self.show_mask_box.value:
            self.select_mask.set_data(self.mask)
            self.select_mask.set_alpha(0.2)
        else:
            self.select_mask.set_alpha(0)
        self.fig.canvas.draw_idle()

    def on_show_origin_change(self, _) -> None:
        if self.is_finished:
            return

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or self.is_finished:
            return

        # 計算靠近滑鼠點擊的點
        self.toggle_near_mask(
            event.xdata,
            event.ydata,
            self.width_slider.value,
            self.mask,
            self.operation_tb.value,
        )

        # 更新 mask
        self.select_mask.set_data(self.mask)

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_perform_all(self, _) -> None:
        if self.is_finished:
            return

        if self.operation_tb.value == "Select":
            self.mask = np.ones_like(self.mask)
        elif self.operation_tb.value == "Erase":
            self.mask = np.zeros_like(self.mask)

        self.select_mask.set_data(self.mask)

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_finish(self, _) -> None:
        self.finish_interactive()

        # also clear the output
        clear_output(wait=False)

    def finish_interactive(self) -> None:
        self.is_finished = True
        plt.close(self.fig)

    def get_positions(self, finish: bool = True) -> Tuple[ndarray, ndarray]:
        if not self.is_finished and finish:
            self.finish_interactive()

        return self.s_mAs, self.s_fpts
