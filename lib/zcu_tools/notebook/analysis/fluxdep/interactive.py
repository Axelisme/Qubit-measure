# Interactive tools for flux-dependent analysis

"""Interactive tools for flux-dependent analysis.

This module provides interactive tools for analyzing flux-dependent spectroscopy data,
including tools for finding points, selecting lines, and selecting points.
"""

from threading import Timer
from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from numpy import ndarray

from zcu_tools.simulate import mA2flx

from .models import energy2transition
from .processing import (
    cast2real_and_norm,
    diff_mirror,
    downsample_points,
    spectrum2d_findpoint,
)


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
        self.show_mask_box.observe(self.on_select_show, names="value")
        self.show_origin_box.observe(self.on_show_origin_change, names="value")
        self.perform_all_bt.on_click(self.on_perform_all)
        self.finish_button.on_click(self.on_finish)

    def init_background(self, spectrum, mAs, fpts) -> None:
        amps = cast2real_and_norm(spectrum)

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

        amps = cast2real_and_norm(spectrum)
        self.s_mAs, self.s_fpts = spectrum2d_findpoint(
            mAs, fpts, amps, threshold, weight=self.mask
        )
        self.scatter = self.ax.scatter(self.s_mAs, self.s_fpts, color="r", s=2)

    def init_callback(self) -> None:
        # 綁定事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

    def update_points(self) -> None:
        threshold = self.threshold_slider.value

        amps = cast2real_and_norm(self.spectrum)
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


class InteractiveLines:
    TRACK_INFO = {
        "red": "<span style='color:red'>正在移動紅線</span>",
        "blue": "<span style='color:blue'>正在移動藍線</span>",
        "none": "<span style='color:gray'>未選擇線條</span>",
    }

    def __init__(self, spectrum, mAs, fpts, mA_c=None, mA_e=None) -> None:
        plt.ioff()  # 避免立即顯示圖表
        self.fig_main, self.ax_main = plt.subplots()
        self.fig_zoom, self.ax_zoom = plt.subplots()
        self.fig_main.tight_layout()
        self.fig_zoom.tight_layout()
        plt.ion()

        # 初始化線的位置
        self.mA_c = (mAs[0] + mAs[-1]) / 2 if mA_c is None else mA_c
        self.mA_e = mAs[-5] if mA_e is None else mA_e

        self.mAs = mAs
        self.fpts = fpts
        self.spectrum = spectrum

        # 新增屬性: 僅使用振幅 (magnitude) 模式，預設為 False (表示同時使用相位資訊)
        self.only_use_magnitude: bool = False

        # 根據 only_use_magnitude 計算顯示用資料
        self.real_signals = cast2real_and_norm(
            spectrum, use_phase=not self.only_use_magnitude
        )

        self.mouse_x = None
        self.mouse_y = None
        self.prev_mouse_x = None
        self.prev_mouse_y = None

        # Flag to確認滑鼠是否真的移動過，用於減少 update_zoom_view 次數
        self._mouse_moved = False

        self.create_widgets()
        self.create_background(mAs, fpts, self.real_signals)
        self.create_lines(mAs)
        self.create_zoom(mAs, fpts, self.real_signals)

        # 顯示 widget
        display(
            widgets.HBox(
                [
                    self.fig_main.canvas,
                    widgets.VBox(
                        [
                            widgets.HBox(
                                [
                                    self.red_button,
                                    self.blue_button,
                                ]
                            ),
                            self.position_text,
                            widgets.HBox(
                                [
                                    self.auto_align_button,
                                    self.swap_button,
                                    self.conjugate_checkbox,
                                    self.only_use_magnitude_checkbox,
                                ]
                            ),
                            widgets.HBox(
                                [
                                    self.status_text,
                                    self.finish_button,
                                ]
                            ),
                            self.fig_zoom.canvas,
                        ]
                    ),
                ]
            )
        )

    def create_widgets(self) -> None:
        """創建 ipywidgets 控件"""
        self.red_button = widgets.Button(
            description="選擇紅線",
            button_style="danger",
            tooltip="選擇紅色線進行移動",
        )
        self.blue_button = widgets.Button(
            description="選擇藍線",
            button_style="info",
            tooltip="選擇藍色線進行移動",
        )
        self.finish_button = widgets.Button(
            description="完成",
            button_style="success",
            tooltip="完成選擇並返回結果",
        )
        self.conjugate_checkbox = widgets.Checkbox(
            value=False, description="Conjugate Line", indent=False
        )

        # 新增: 僅使用振幅顯示的切換開關
        self.only_use_magnitude_checkbox = widgets.Checkbox(
            value=self.only_use_magnitude,
            description="Magnitude Only",
            indent=False,
        )
        self.swap_button = widgets.Button(
            description="交換線條",
            button_style="warning",
            tooltip="交換紅線與藍線的位置",
        )
        self.auto_align_button = widgets.Button(
            description="自動對齊",
            button_style="primary",
            tooltip="自動找到mirror loss最小的位置並對齊線條",
        )
        self.position_text = widgets.HTML(value=self.get_info())
        self.status_text = widgets.HTML(
            value="<span style='color:gray'>未選擇線條</span>"
        )

        # 綁定事件
        self.red_button.on_click(self.set_picked_red)
        self.blue_button.on_click(self.set_picked_blue)
        self.finish_button.on_click(self.on_finish)
        self.swap_button.on_click(self.swap_lines)
        self.auto_align_button.on_click(self.auto_align_lines)

        # 綁定 magnitude 顯示模式切換
        self.only_use_magnitude_checkbox.observe(
            self.on_toggle_magnitude, names="value"
        )

    def create_background(self, mAs, fpts, amps) -> None:
        """創建背景圖片"""
        # 儲存圖像控制柄以便之後更新資料
        dx = (mAs[-1] - mAs[0]) / (len(mAs) - 1)
        dy = (fpts[-1] - fpts[0]) / (len(fpts) - 1)
        self.main_im = self.ax_main.imshow(
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

        self.ax_main.set_xlim(mAs[0], mAs[-1])
        self.ax_main.set_ylim(fpts[0], fpts[-1])

    def create_lines(self, mAs) -> None:
        """創建兩條垂直線"""
        # 創建兩條垂直線
        self.rline = self.ax_main.axvline(x=self.mA_c, color="r", linestyle="--")
        self.bline = self.ax_main.axvline(x=self.mA_e, color="b", linestyle="--")

        # 設置變數
        self.picked = None
        self.min_dist = 0.1 * (mAs[-1] - mAs[0])
        self.is_finished = False
        self.active_line = None  # 用來跟踪目前正在移動的線

        # 連接事件
        self.fig_main.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig_main.canvas.mpl_connect("motion_notify_event", self.onmove)

        # 創建動畫
        self.anim_main = FuncAnimation(
            self.fig_main,
            self.update_main_view,
            interval=33,  # 約30 FPS
            blit=True,
            cache_frame_data=False,
        )

    def create_zoom(self, mAs, fpts, amps) -> None:
        """創建放大視圖"""
        self.ax_zoom.set_title(f"mirror loss: {None}")

        dx = (mAs[-1] - mAs[0]) / (len(mAs) - 1)
        dy = (fpts[-1] - fpts[0]) / (len(fpts) - 1)
        self.zoom_im = self.ax_zoom.imshow(
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
        self.ax_zoom.set_xticks([])
        self.ax_zoom.set_yticks([])

        # show red spot at center
        x = self.mA_c
        y = 0.5 * (fpts[0] + fpts[-1])
        self.zoom_dot = self.ax_zoom.plot([x], [y], "ro")[0]

        self.anim_zoom = FuncAnimation(
            self.fig_zoom,
            self.update_zoom_view,
            interval=100,
            blit=True,
            cache_frame_data=False,
        )

    def get_info(self) -> str:
        return f"紅線: {self.mA_c:.2e}, 藍線: {self.mA_e:.2e}, 週期：{2 * abs(self.mA_e - self.mA_c):.2e}"

    def set_picked_red(self, _) -> None:
        """選擇紅線"""
        if self.is_finished:
            return

        if self.active_line == self.rline:
            # 如果已經在移動紅線, 則停止移動
            self.stop_tracking()
        else:
            # 開始移動紅線
            self.active_line = self.rline
            self.picked = self.rline
            self.status_text.value = self.TRACK_INFO["red"]

    def set_picked_blue(self, _) -> None:
        """選擇藍線"""
        if self.is_finished:
            return

        if self.active_line == self.bline:
            # 如果已經在移動藍線, 則停止移動
            self.stop_tracking()
        else:
            # 開始移動藍線
            self.active_line = self.bline
            self.picked = self.bline
            self.status_text.value = self.TRACK_INFO["blue"]

    def stop_tracking(self) -> None:
        """停止追蹤滑鼠"""
        self.active_line = None
        self.picked = None
        self.status_text.value = self.TRACK_INFO["none"]

    def swap_lines(self, _) -> None:
        """交換紅線與藍線的位置"""
        if self.is_finished:
            return

        # 停止當前的追蹤
        self.stop_tracking()

        # 交換線條的位置
        temp_mA_c = self.mA_c
        self.mA_c = self.mA_e
        self.mA_e = temp_mA_c

        # 更新線條的視覺位置
        self.rline.set_xdata([self.mA_c])
        self.bline.set_xdata([self.mA_e])

        # 更新位置文字
        self.position_text.value = self.get_info()

        # 重新繪製
        self.fig_main.canvas.draw_idle()

    def auto_align_lines(self, _) -> None:
        """自動對齊線條到mirror loss最小的位置"""
        if self.is_finished:
            return

        # 停止當前的追蹤
        self.stop_tracking()

        # 計算搜索範圍：總寬度的二十分之一
        total_width = self.mAs[-1] - self.mAs[0]
        search_width = total_width / 20

        # 為紅線和藍線分別找到最佳位置
        best_red_pos = self._find_best_position(self.mA_c, search_width)
        best_blue_pos = self._find_best_position(self.mA_e, search_width)

        # 更新線條位置
        self.mA_c = best_red_pos
        self.mA_e = best_blue_pos

        # 更新線條的視覺位置
        self.rline.set_xdata([self.mA_c])
        self.bline.set_xdata([self.mA_e])

        # 更新位置文字
        self.position_text.value = self.get_info()

        # 重新繪製
        self.fig_main.canvas.draw_idle()

    def _find_best_position(self, current_pos: float, search_width: float) -> float:
        """在給定範圍內找到mirror loss最小的位置"""
        # 計算半格精度
        precision = 0.25 * (self.mAs[-1] - self.mAs[0]) / len(self.mAs)

        # 定義搜索範圍
        left_bound = max(self.mAs[0], current_pos - search_width / 2)
        right_bound = min(self.mAs[-1], current_pos + search_width / 2)

        # 創建候選位置，只考慮半格的整數倍
        # 將邊界轉換為相對於起始點的格數
        left_steps = int(np.floor((left_bound - self.mAs[0]) / precision))
        right_steps = int(np.ceil((right_bound - self.mAs[0]) / precision))

        # 生成候選位置
        candidates = [
            self.mAs[0] + i * precision for i in range(left_steps, right_steps + 1)
        ]

        # 確保候選位置在有效範圍內
        candidates = [pos for pos in candidates if self.mAs[0] <= pos <= self.mAs[-1]]

        # 如果沒有有效的候選位置，返回當前位置
        if not candidates:
            return current_pos

        best_pos = current_pos
        min_loss = float("inf")

        # 對每個候選位置計算mirror loss
        for candidate in candidates:
            # 計算該位置的mirror loss
            # 總是使用spectrum來計算(包含phase資訊)
            diff_amps = diff_mirror(self.mAs, self.spectrum.T, candidate).T
            valid_amps = diff_amps[diff_amps != 0.0]

            # 確保有有效的數據點
            if len(valid_amps) > 0:
                mirror_loss = np.mean(valid_amps)

                # 如果這個位置的loss更小，更新最佳位置
                if not np.isnan(mirror_loss) and mirror_loss < min_loss:
                    min_loss = mirror_loss
                    best_pos = candidate

        return best_pos

    def onclick(self, event) -> None:
        """滑鼠點擊事件"""
        if self.is_finished or event.inaxes != self.ax_main:
            return

        # 判斷點擊了哪條線
        red_x = self.rline.get_xdata()[0]
        blue_x = self.bline.get_xdata()[0]

        red_dist = abs(event.xdata - red_x)
        blue_dist = abs(event.xdata - blue_x)

        # 如果已經有活動的線條, 點擊任何位置都停止追蹤
        if self.active_line is not None:
            self.stop_tracking()
            return

        # 選擇最近的線
        if red_dist < blue_dist and red_dist < self.min_dist / 2:
            self.set_picked_red(None)
        elif blue_dist < red_dist and blue_dist < self.min_dist / 2:
            self.set_picked_blue(None)

    def onmove(self, event) -> None:
        """滑鼠移動事件"""
        if self.is_finished or event.inaxes != self.ax_main:
            self.mouse_x = None
            self.mouse_y = None
            return
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

        # 標記滑鼠有移動，供 update_zoom_view 使用
        self._mouse_moved = True

        # 更新前一次位置紀錄
        self.prev_mouse_x = self.mouse_x
        self.prev_mouse_y = self.mouse_y

    def update_main_view(self, _) -> list:
        """更新動畫"""
        x, y = self.mouse_x, self.mouse_y
        if self.picked is None or x is None or y is None:
            return []

        other_line = self.bline if self.picked is self.rline else self.rline
        other_x = other_line.get_xdata()[0]

        # 確保線之間保持最小距離
        if x > other_x and x - other_x < self.min_dist:
            x = other_x + self.min_dist
        elif x < other_x and other_x - x < self.min_dist:
            x = other_x - self.min_dist

        # 更新線的位置
        if self.conjugate_checkbox.value:
            # 同步移動
            dx = x - self.picked.get_xdata()[0]
            # 更新兩條線的位置
            self.rline.set_xdata([self.rline.get_xdata()[0] + dx])
            self.bline.set_xdata([self.bline.get_xdata()[0] + dx])
        else:
            # 單獨移動
            self.picked.set_xdata([x])

        # 更新位置文字
        self.mA_c = self.rline.get_xdata()[0]
        self.mA_e = self.bline.get_xdata()[0]
        self.position_text.value = self.get_info()

        return [self.rline, self.bline]

    def update_zoom_view(self, _) -> list:
        """更新放大視圖"""
        x, y = self.mouse_x, self.mouse_y

        # 條件不足或滑鼠未移動時不更新，減少 CPU loading
        if x is None or y is None or self.active_line is None or not self._mouse_moved:
            return []  # do nothing

        # reset flag
        self._mouse_moved = False

        # 總是使用spectrum來計算(包含phase資訊)
        diff_amps = diff_mirror(self.mAs, self.spectrum.T, x).T
        self.zoom_im.set_data(diff_amps)

        mirror_loss = np.mean(diff_amps[diff_amps != 0.0])

        # set axis limits to simulate zoom
        Dx = 0.3 * (self.mAs[-1] - self.mAs[0])
        Dy = 0.3 * (self.fpts[-1] - self.fpts[0])
        self.ax_zoom.set_xlim(x - Dx, x + Dx)
        self.ax_zoom.set_ylim(y - Dy, y + Dy)
        self.ax_zoom.set_title(f"mirror loss: {mirror_loss:.4f}")

        self.zoom_dot.set_xdata([x])
        self.zoom_dot.set_ydata([y])
        self.zoom_dot.set_color("r" if self.active_line is self.rline else "b")

        return [self.zoom_im, self.zoom_dot]

    def on_finish(self, _) -> None:
        """完成按鈕的回調函數"""
        self.finish_interactive()

        # also clear the output
        clear_output(wait=False)

    def finish_interactive(self) -> None:
        self.is_finished = True
        self.picked = None
        self.active_line = None
        # 停止動畫
        self.anim_main.event_source.stop()
        self.anim_zoom.event_source.stop()
        # plt.close(self.fig_main)
        # plt.close(self.fig_zoom)

    def get_positions(self, finish: bool = True) -> Tuple[float, float]:
        """運行交互式選擇器並返回兩條線的位置"""
        if not self.is_finished and finish:
            self.finish_interactive()
        precision = 0.5 * (self.mAs[-1] - self.mAs[0]) / len(self.mAs)
        mA_c = precision * round((self.mA_c - self.mAs[0]) / precision) + self.mAs[0]
        mA_e = precision * round((self.mA_e - self.mAs[0]) / precision) + self.mAs[0]
        return float(mA_c), float(mA_e)

    # ----------------------------------
    #  Event handlers
    # ----------------------------------

    def on_toggle_magnitude(self, change) -> None:
        """切換是否僅使用振幅資料的顯示模式"""
        if self.is_finished:
            return

        # 停止當前的追蹤
        self.stop_tracking()

        # 更新屬性
        self.only_use_magnitude = bool(change["new"])

        # 重新計算要顯示的資料
        self.real_signals = cast2real_and_norm(
            self.spectrum, use_phase=not self.only_use_magnitude
        )

        self.main_im.set_data(self.real_signals)
        self.main_im.autoscale()

        # 更新背景影像
        self.fig_main.canvas.draw_idle()


class InteractiveSelector:
    def __init__(self, s_pects, selected=None, brush_width=0.05) -> None:
        self.s_spects = s_pects

        self.s_mAs = np.concatenate([s["points"]["mAs"] for s in s_pects.values()])
        self.s_fpts = np.concatenate([s["points"]["fpts"] for s in s_pects.values()])

        self.selected = (
            selected if selected is not None else np.ones_like(self.s_mAs, dtype=bool)
        )
        self.filter_mask = np.ones_like(self.selected, dtype=bool)

        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        plt.ion()

        self.set_plot_limit()
        self.create_widgets(brush_width)
        self.init_background(s_pects)
        self.init_points(self.s_mAs, self.s_fpts, self.selected)

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

        sel_x = self.s_mAs[self.selected] / (self.mA_bound[1] - self.mA_bound[0])
        sel_y = self.s_fpts[self.selected] / (self.fpt_bound[1] - self.fpt_bound[0])
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

    def init_background(self, s_pects) -> None:
        for spect in s_pects.values():
            # Get corresponding data and range
            signals = spect["spectrum"]["data"] ** 1.5
            flx_mask = np.any(~np.isnan(signals), axis=0)
            fpt_mask = np.any(~np.isnan(signals), axis=1)
            signals = signals[fpt_mask, :][:, flx_mask]

            # Normalize data
            amps = cast2real_and_norm(signals)

            # Add heatmap trace
            sp_mAs = spect["spectrum"]["mAs"][flx_mask]
            sp_fpts = spect["spectrum"]["fpts"][fpt_mask]
            self.ax.imshow(
                amps,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(sp_mAs[0], sp_mAs[-1], sp_fpts[0], sp_fpts[-1]),
            )

    def init_points(self, s_mAs, s_fpts, selected) -> None:
        self.scatter = self.ax.scatter(
            s_mAs, s_fpts, c=selected.astype(float), s=2, vmax=1, vmin=0
        )

    def set_plot_limit(self) -> None:
        spect_list = [s["spectrum"] for s in self.s_spects.values()]
        sp_mAs = np.concatenate([s["mAs"] for s in spect_list])
        sp_fpts = np.concatenate([s["fpts"] for s in spect_list])
        self.mA_bound = (
            min(np.nanmin(sp_mAs), self.s_mAs.min()),
            max(np.nanmax(sp_mAs), self.s_mAs.max()),
        )
        self.fpt_bound = (
            min(np.nanmin(sp_fpts), self.s_fpts.min()),
            max(np.nanmax(sp_fpts), self.s_fpts.max()),
        )

        # Set x and y axis range
        self.ax.set_xlim(self.mA_bound[0], self.mA_bound[1])
        self.ax.set_ylim(self.fpt_bound[0], self.fpt_bound[1])

    def get_cur_selected(self) -> np.ndarray:
        cur_selected = np.zeros_like(self.selected)
        cur_selected[np.where(self.selected)[0][self.filter_mask]] = True
        return cur_selected

    def update_points(self) -> None:
        self.scatter.set_array(self.get_cur_selected().astype(float))

    def toggle_near_mask(self, x, y, width):
        x_d = np.abs(self.s_mAs - x) / (self.mA_bound[1] - self.mA_bound[0])
        y_d = np.abs(self.s_fpts - y) / (self.fpt_bound[1] - self.fpt_bound[0])
        toggle_mask = x_d**2 + y_d**2 <= width**2

        self.selected[toggle_mask] = self.operation_tb.value == "Select"

    def on_finish(self, _) -> None:
        self.finish_interactive()

        # also clear the output
        clear_output(wait=False)

    def finish_interactive(self) -> None:
        self.is_finished = True
        plt.close(self.fig)

    def get_positions(self, finish: bool = True) -> Tuple[ndarray, ndarray, np.ndarray]:
        if not self.is_finished and finish:
            self.finish_interactive()

        cur_selected = self.get_cur_selected()
        return (self.s_mAs[cur_selected], self.s_fpts[cur_selected], cur_selected)

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
        x_range = self.mA_bound[1] - self.mA_bound[0]
        y_range = self.fpt_bound[1] - self.fpt_bound[0]

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


class VisualizeSpet:
    def __init__(
        self, s_spects, s_mAs, s_fpts, mAs, energies, allows, auto_hide=False
    ) -> None:
        self.s_spects = s_spects
        self.s_mAs = s_mAs
        self.s_fpts = s_fpts
        self.mAs = mAs
        self.energies = energies
        self.allows = allows

        self.auto_hide = auto_hide  # 新增參數，預設為 False

        # Default scatter point styling
        self.scatter_size = 3
        self.scatter_color = "red"
        self.scatter_color_array = None  # 用於存儲顏色陣列

        if len(self.s_spects) == 0:
            raise ValueError("No spectrum data provided")

        first_spect = next(iter(self.s_spects.values()))
        mA_c, period = first_spect["mA_c"], first_spect["period"]
        self.s_flxs = mA2flx(self.s_mAs, mA_c, period)
        self.flxs = mA2flx(self.mAs, mA_c, period)

        self.mA_c = mA_c
        self.period = period

    def set_scatter_style(self, size=None, color=None) -> "VisualizeSpet":
        if size is not None:
            self.scatter_size = size
        if color is not None:
            # 可以是顏色名稱或數值陣列
            self.scatter_color = color
            # 檢查是否為陣列
            if isinstance(color, (list, np.ndarray)):
                self.scatter_color_array = color

        return self  # For method chaining

    def plot_background(self, fig) -> None:
        # Add heatmap traces for each spectrum in s_spects
        for spect in self.s_spects.values():
            # Get corresponding data and range
            signals = spect["spectrum"]["data"] ** 1.5
            flx_mask = np.any(~np.isnan(signals), axis=0)
            fpt_mask = np.any(~np.isnan(signals), axis=1)
            signals = signals[fpt_mask, :][:, flx_mask]

            # Normalize data
            amps = cast2real_and_norm(signals)

            # Add heatmap trace
            fig.add_trace(
                go.Heatmap(
                    z=amps,
                    x=spect["spectrum"]["mAs"][flx_mask],
                    y=spect["spectrum"]["fpts"][fpt_mask],
                    colorscale="Greys",
                    showscale=False,
                )
            )

    def plot_predict_lines(self, fig) -> None:
        # Calculate transitions
        fs, labels = energy2transition(self.energies, self.allows)

        # 計算哪些線需要隱藏
        visible_lines = self._filter_nearby_lines(fs, self.mAs, self.s_fpts, self.s_mAs)

        # Add transition line traces
        for i, label in enumerate(labels):
            visible = "legendonly" if not visible_lines[i] else True
            fig.add_trace(
                go.Scatter(
                    x=self.mAs,
                    y=fs[:, i],
                    mode="lines",
                    name=label,
                    visible=visible,
                )
            )

    def plot_scatter_point(self, fig) -> None:
        marker_dict = {"size": self.scatter_size}

        # 處理顏色設置
        if self.scatter_color_array is not None:
            # 如果提供了顏色陣列，使用它並指定顏色範圍
            marker_dict["color"] = self.scatter_color_array
            marker_dict["colorscale"] = "Viridis"
            marker_dict["showscale"] = True  # 顯示顏色刻度
            marker_dict["colorbar"] = dict(
                x=-0.1,  # 將顏色條放置在左側
                xanchor="left",  # 錨點在左側
            )
            hovertext = self.scatter_color_array
        else:
            # 否則使用單一顏色
            marker_dict["color"] = self.scatter_color
            hovertext = None

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=self.s_mAs,
                y=self.s_fpts,
                mode="markers",
                marker=marker_dict,
                hovertext=hovertext,
            )
        )

    def plot_constant_freqs(self, fig) -> None:
        if "r_f" in self.allows:
            fig.add_hline(y=self.allows["r_f"], line_dash="dash", name="r_f")

        if "sample_f" in self.allows:
            fig.add_hline(y=self.allows["sample_f"], line_dash="dash", name="sample_f")

    def _set_axis_limit(self, fig) -> None:
        sp_mAs = np.concatenate([s["spectrum"]["mAs"] for s in self.s_spects.values()])
        sp_fpts = np.concatenate(
            [s["spectrum"]["fpts"] for s in self.s_spects.values()]
        )
        mA_bound = (
            min(np.nanmin(sp_mAs), self.mAs.min()),
            max(np.nanmax(sp_mAs), self.mAs.max()),
        )
        fpt_bound = (
            min(
                np.nanmin(sp_fpts),
                self.allows.get("r_f", np.inf) - 0.1,
                self.allows.get("sample_f", np.inf) - 0.1,
            ),
            max(
                np.nanmax(sp_fpts),
                self.allows.get("r_f", 0.0) + 0.1,
                self.allows.get("sample_f", 0.0) + 0.1,
            ),
        )

        if len(self.s_fpts) > 0:
            mA_bound = (
                min(mA_bound[0], self.s_mAs.min()),
                max(mA_bound[1], self.s_mAs.max()),
            )
            fpt_bound = (
                min(fpt_bound[0], self.s_fpts.min()),
                max(fpt_bound[1], self.s_fpts.max()),
            )

        # Set x and y axis range
        fig.update_xaxes(range=[mA_bound[0], mA_bound[1]])
        fig.update_yaxes(range=[0.0, fpt_bound[1]])

    def create_figure(self) -> go.Figure:
        fig = go.Figure()

        self.plot_background(fig)
        self.plot_predict_lines(fig)
        self.plot_constant_freqs(fig)
        self.plot_scatter_point(fig)

        self._set_axis_limit(fig)

        # Secondary x axis, show flxs
        fig.add_scatter(
            x=self.flxs, y=np.zeros_like(self.flxs), xaxis="x2", opacity=0.0
        )
        ticks_mAs = self.mAs[:: max(1, len(self.mAs) // 20)]
        ticks_flxs = self.flxs[:: max(1, len(self.flxs) // 20)]
        fig.update_layout(
            xaxis2=dict(
                tickvals=ticks_mAs,
                ticktext=[f"{flx:.2f}" for flx in ticks_flxs],
                matches="x1",
                overlaying="x1",
                side="top",
            )
        )

        # Update layout
        fig.update_layout(
            margin=dict(t=100),
            legend_title_text="Transition",
            title_x=0.5,
            xaxis_title="mAs",
            yaxis_title="Frequency (GHz)",
            legend=dict(x=1, y=0.5),
            height=1600,
        )

        return fig

    def _filter_nearby_lines(self, fs, mAs, s_fpts, s_mAs) -> np.ndarray:
        """
        計算哪些轉換線靠近散點，並返回布林陣列，決定要顯示哪些線。

        Parameters:
        fs: numpy array, 所有轉換線的頻率數據 (M, K)
        mAs: numpy array, 所有轉換線對應的通量數據 (M, )
        s_fpts: numpy array, 所有散點的頻率數據 (N, )
        s_mAs: numpy array, 所有散點的通量數據 (N, )

        Returns:
        visible_lines: numpy array, 形狀 (K, ), True 表示該線要顯示, False 表示要隱藏
        """
        K = fs.shape[1]

        THRESHOLD = 4

        if self.auto_hide:
            # interpolate flux points
            s_fs = np.array(
                [np.interp(s_mAs, mAs, fs[:, i]) for i in range(fs.shape[1])]
            ).T  # (N, K)

            # 計算散點與所有線之間的距離
            dists = np.abs(s_fs - s_fpts[:, None])  # (N, K)
            matchs = np.argmin(dists, axis=1)  # (N, )

            # if only one or two points are matched this line, make it invisible
            visible_lines = (
                np.sum(matchs[:, None] == np.arange(K)[None, :], axis=0) > THRESHOLD
            )
        else:
            visible_lines = np.full(K, True)

        return visible_lines
