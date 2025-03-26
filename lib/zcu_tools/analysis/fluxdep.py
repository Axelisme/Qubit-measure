import json
import os
from threading import Timer
from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from h5py import File
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from numba import njit
from scipy.signal import find_peaks
from tqdm.auto import tqdm, trange

from zcu_tools.tools import AsyncFunc


class InteractiveFindPoints:
    def __init__(self, spectrum, flxs, fpts, threshold=1.0, brush_width=0.05):
        self.spectrum = spectrum
        self.flxs = flxs
        self.fpts = fpts

        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        plt.ion()

        # 顯示 widget
        self.create_widgets(threshold, brush_width)

        # 顯示頻譜
        self.init_background(spectrum, flxs, fpts)

        # 顯示 mask
        self.init_mask(fpts, flxs)

        # 顯示發現的點
        self.init_points(flxs, fpts, spectrum)

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

    def create_widgets(self, threshold, brush_width):
        self.threshold_slider = widgets.FloatSlider(
            value=threshold, min=1.0, max=20.0, step=0.01, description="Threshold:"
        )
        self.width_slider = widgets.FloatSlider(
            value=brush_width, min=0.01, max=0.1, step=1e-4, description="Brush Width:"
        )
        self.show_mask_box = widgets.Checkbox(value=False, description="Show Mask")
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
        self.perform_all_bt.on_click(self.on_perform_all)
        self.finish_button.on_click(self.on_finish)

    def init_background(self, spectrum, flxs, fpts):
        s_spectrum = np.abs(spectrum - np.mean(spectrum, axis=0, keepdims=True))
        s_spectrum /= np.std(s_spectrum, axis=0, keepdims=True)
        self.spectrum_img = self.ax.imshow(
            s_spectrum,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        )

    def init_mask(self, fpts, flxs):
        self.mask = np.ones((len(fpts), len(flxs)), dtype=bool)

        self.select_mask = self.ax.imshow(
            self.mask,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
            alpha=0.2 if self.show_mask_box.value else 0,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

    def init_points(self, flxs, fpts, spectrum):
        threshold = self.threshold_slider.value
        self.s_flxs, self.s_fpts = spectrum_analyze(
            flxs, fpts, spectrum, threshold, weight=self.mask
        )
        self.scatter = self.ax.scatter(self.s_flxs, self.s_fpts, color="r", s=2)

    def init_callback(self):
        # 綁定事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

    def update_points(self):
        threshold = self.threshold_slider.value
        self.s_flxs, self.s_fpts = spectrum_analyze(
            self.flxs, self.fpts, self.spectrum, threshold, weight=self.mask
        )
        self.scatter.set_offsets(np.column_stack((self.s_flxs, self.s_fpts)))

    def toggle_near_mask(self, x, y, width, mask, mode):
        x_d = np.abs(self.flxs - x) / (self.flxs[-1] - self.flxs[0])
        y_d = np.abs(self.fpts - y) / (self.fpts[-1] - self.fpts[0])
        d2 = x_d[None, :] ** 2 + y_d[:, None] ** 2

        weight = d2 <= width**2
        if mode == "Select":
            mask |= weight
        elif mode == "Erase":
            mask &= ~weight

    def on_ratio_change(self, _):
        if self.is_finished:
            return

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_select_show(self, _):
        if self.is_finished:
            return

        if self.show_mask_box.value:
            self.select_mask.set_data(self.mask)
            self.select_mask.set_alpha(0.2)
        else:
            self.select_mask.set_alpha(0)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
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

    def on_perform_all(self, _):
        if self.is_finished:
            return

        if self.operation_tb.value == "Select":
            self.mask = np.ones_like(self.mask)
        elif self.operation_tb.value == "Erase":
            self.mask = np.zeros_like(self.mask)

        self.select_mask.set_data(self.mask)

        self.update_points()
        self.fig.canvas.draw_idle()

    def on_finish(self, _):
        plt.close(self.fig)
        self.is_finished = True

    def get_positions(self):
        if not self.is_finished:
            self.on_finish(None)
        return self.s_flxs, self.s_fpts


class InteractiveLines:
    TRACK_INFO = {
        "red": "<span style='color:red'>正在移動紅線</span>",
        "blue": "<span style='color:blue'>正在移動藍線</span>",
        "none": "<span style='color:gray'>未選擇線條</span>",
    }

    def __init__(self, spectrum, flxs, fpts, cflx=None, eflx=None):
        plt.ioff()  # 避免立即顯示圖表
        self.fig_main, self.ax_main = plt.subplots(num=None)
        self.fig_zoom, self.ax_zoom = plt.subplots(figsize=(5, 5), num=None)
        self.fig_main.tight_layout()
        self.fig_zoom.tight_layout()
        plt.ion()

        # 初始化線的位置
        self.cflx = (flxs[0] + flxs[-1]) / 2 if cflx is None else cflx
        self.eflx = flxs[-5] if eflx is None else eflx

        self.flxs = flxs
        self.fpts = fpts
        self.spectrum = spectrum

        self.mouse_x = None
        self.mouse_y = None

        self.create_widgets()
        self.create_background(flxs, fpts, spectrum)
        self.create_lines(flxs)
        self.create_zoom(flxs, fpts, spectrum)

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

    def create_widgets(self):
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
        self.position_text = widgets.HTML(value=self.get_info())
        self.status_text = widgets.HTML(
            value="<span style='color:gray'>未選擇線條</span>"
        )

        # 綁定事件
        self.red_button.on_click(self.set_picked_red)
        self.blue_button.on_click(self.set_picked_blue)
        self.finish_button.on_click(self.on_finish)

    def create_background(self, flxs, fpts, spectrum):
        """創建背景圖片"""
        # 顯示光譜圖
        self.ax_main.imshow(
            spectrum,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        )

        # xlim, ylim
        self.ax_main.set_xlim(flxs[0], flxs[-1])
        self.ax_main.set_ylim(fpts[0], fpts[-1])

    def create_lines(self, flxs):
        """創建兩條垂直線"""
        # 創建兩條垂直線
        self.rline = self.ax_main.axvline(x=self.cflx, color="r", linestyle="--")
        self.bline = self.ax_main.axvline(x=self.eflx, color="b", linestyle="--")

        # 設置變數
        self.picked = None
        self.min_dist = 0.1 * (flxs[-1] - flxs[0])
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

    def create_zoom(self, flxs, fpts, spectrum):
        """創建放大視圖"""
        self.ax_zoom.set_title("Zoom View")
        self.zoom_im = self.ax_zoom.imshow(
            spectrum,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        )
        self.ax_zoom.set_xticks([])
        self.ax_zoom.set_yticks([])

        # show red spot at center
        x = self.cflx
        y = 0.5 * (fpts[0] + fpts[-1])
        self.zoom_dot = self.ax_zoom.plot([x], [y], "ro")[0]

        self.anim_zoom = FuncAnimation(
            self.fig_zoom,
            self.update_zoom_view,
            interval=33,
            blit=True,
            cache_frame_data=False,
        )

    def get_info(self):
        return f"紅線: {self.cflx:.2e}, 藍線: {self.eflx:.2e}, 週期：{2 * abs(self.eflx - self.cflx):.2e}"

    def set_picked_red(self, _):
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

    def set_picked_blue(self, _):
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

    def stop_tracking(self):
        """停止追蹤滑鼠"""
        self.active_line = None
        self.picked = None
        self.status_text.value = self.TRACK_INFO["none"]

    def onclick(self, event):
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

    def onmove(self, event):
        """滑鼠移動事件"""
        if self.is_finished or event.inaxes != self.ax_main:
            self.mouse_x = None
            self.mouse_y = None
            return
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

    def update_main_view(self, _):
        """更新動畫"""
        if self.picked is None or self.mouse_x is None:
            return [self.rline, self.bline]

        new_x = self.mouse_x
        other_line = self.bline if self.picked is self.rline else self.rline
        other_x = other_line.get_xdata()[0]

        # 確保線之間保持最小距離
        if new_x > other_x and new_x - other_x < self.min_dist:
            new_x = other_x + self.min_dist
        elif new_x < other_x and other_x - new_x < self.min_dist:
            new_x = other_x - self.min_dist

        # 確保不超出邊界
        if new_x > self.flxs[-1]:
            new_x = self.flxs[-1]
        elif new_x < self.flxs[0]:
            new_x = self.flxs[0]

        # 更新線的位置
        self.picked.set_xdata([new_x, new_x])

        # 更新位置文字
        self.cflx = self.rline.get_xdata()[0]
        self.eflx = self.bline.get_xdata()[0]
        self.position_text.value = self.get_info()

        return [self.rline, self.bline]

    def update_zoom_view(self, _):
        """更新放大視圖"""
        x, y = self.mouse_x, self.mouse_y
        if x is None or y is None or self.active_line is None:
            return []  # out of axes or not dragging, do nothing

        # set axis limits to simulate zoom
        Dx = 0.1 * (self.flxs[-1] - self.flxs[0])
        Dy = 0.1 * (self.fpts[-1] - self.fpts[0])
        self.ax_zoom.set_xlim(x - Dx, x + Dx)
        self.ax_zoom.set_ylim(y - Dy, y + Dy)

        self.zoom_dot.set_xdata([x])
        self.zoom_dot.set_ydata([y])
        self.zoom_dot.set_color("r" if self.active_line is self.rline else "b")

        return [self.zoom_im, self.zoom_dot]

    def on_finish(self, _):
        """完成按鈕的回調函數"""
        self.is_finished = True
        self.picked = None
        self.active_line = None
        # 停止動畫
        self.anim_main.event_source.stop()
        self.anim_zoom.event_source.stop()
        plt.close(self.fig_main)
        plt.close(self.fig_zoom)

    def get_positions(self):
        """運行交互式選擇器並返回兩條線的位置"""
        if not self.is_finished:
            self.on_finish(None)
        return float(self.cflx), float(self.eflx)


class InteractiveSelector:
    def __init__(self, s_pects, selected=None, brush_width=0.05):
        self.s_spects = s_pects

        self.s_flxs = np.concatenate(
            [
                self.get_flxs(s["points"]["mAs"], s["mA_c"], s["period"])
                for s in s_pects.values()
            ]
        )
        self.s_fpts = np.concatenate([s["points"]["fpts"] for s in s_pects.values()])
        self.selected = (
            selected if selected is not None else np.ones_like(self.s_flxs, dtype=bool)
        )

        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        plt.ion()

        # 顯示 widget
        self.create_widgets(brush_width)

        # 顯示頻譜
        self.init_background(s_pects)

        # 顯示發現的點
        self.init_points(self.s_flxs, self.s_fpts, self.selected)

        # 設置 x 和 y 軸範圍
        self.set_plot_limit()

        # 準備手繪曲線
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

        display(
            widgets.HBox(
                [
                    self.fig.canvas,
                    widgets.VBox(
                        [self.width_slider, self.operation_tb, self.finish_button]
                    ),
                ]
            )
        )

        # 新增: 用於儲存暫時性圓圈和計時器的變數
        self.temp_circle = None
        self.temp_circle_timer = None

    def get_flxs(self, mAs, mA_c, period):
        return (mAs - mA_c) / period

    def create_widgets(self, brush_width):
        self.width_slider = widgets.FloatSlider(
            value=brush_width, min=0.01, max=0.1, step=1e-4, description="Brush Width:"
        )
        self.operation_tb = widgets.Dropdown(
            options=["Select", "Erase"], value="Select", description="Operation:"
        )
        self.finish_button = widgets.Button(
            description="Finish", button_style="success"
        )
        self.finish_button.on_click(self.on_finish)

    def init_background(self, s_pects):
        for spect in s_pects.values():
            # Get corresponding data and range
            data = spect["spectrum"]["data"] ** 1.5
            flx_mask = np.any(~np.isnan(data), axis=0)
            fpt_mask = np.any(~np.isnan(data), axis=1)
            data = data[fpt_mask, :][:, flx_mask]

            # Normalize data
            data = np.abs(data - np.mean(data, axis=0, keepdims=True))
            data /= np.std(data, axis=0, keepdims=True)

            # Add heatmap trace
            sp_flxs = self.get_flxs(
                spect["spectrum"]["mAs"][flx_mask], spect["mA_c"], spect["period"]
            )
            sp_fpts = spect["spectrum"]["fpts"][fpt_mask]
            self.ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(sp_flxs[0], sp_flxs[-1], sp_fpts[0], sp_fpts[-1]),
            )

    def init_points(self, flxs, s_fpts, selected):
        self.scatter = self.ax.scatter(
            flxs, s_fpts, c=selected.astype(float), s=2, vmax=1, vmin=0
        )

    def set_plot_limit(self):
        sp_flxs = np.concatenate(
            [
                self.get_flxs(s["points"]["mAs"], s["mA_c"], s["period"])
                for s in self.s_spects.values()
            ]
        )
        sp_fpts = np.concatenate([s["points"]["fpts"] for s in self.s_spects.values()])
        self.flxs_bound = (
            min(np.nanmin(sp_flxs), self.s_flxs.min()),
            max(np.nanmax(sp_flxs), self.s_flxs.max()),
        )
        self.fpt_bound = (
            min(np.nanmin(sp_fpts), self.s_fpts.min()),
            max(np.nanmax(sp_fpts), self.s_fpts.max()),
        )

        # Set x and y axis range
        self.ax.set_xlim(self.flxs_bound[0], self.flxs_bound[1])
        self.ax.set_ylim(self.fpt_bound[0], self.fpt_bound[1])

    def update_points(self, selected):
        self.scatter.set_array(selected.astype(float))

    def toggle_near_mask(self, x, y, width):
        x_d = np.abs(self.s_flxs - x) / (self.flxs_bound[1] - self.flxs_bound[0])
        y_d = np.abs(self.s_fpts - y) / (self.fpt_bound[1] - self.fpt_bound[0])
        toggle_mask = x_d**2 + y_d**2 <= width**2

        self.selected[toggle_mask] = self.operation_tb.value == "Select"

    def on_finish(self, _):
        plt.close(self.fig)
        self.is_finished = True

    def get_positions(self):
        if not self.is_finished:
            self.on_finish(None)

        return self.s_flxs[self.selected], self.s_fpts[self.selected], self.selected

    def on_press(self, event):
        if event.inaxes != self.ax or self.is_finished:
            return

        # 計算靠近滑鼠點擊的點
        width = self.width_slider.value
        self.toggle_near_mask(event.xdata, event.ydata, width)

        # 新增: 顯示暫時性圓圈（並取消舊的計時器）
        self.show_temp_circle(event.xdata, event.ydata, width)

        self.update_points(self.selected)
        self.fig.canvas.draw_idle()

    def show_temp_circle(self, x, y, width):
        """顯示暫時性圓圈，一秒後消失"""
        # 移除現有的暫時性圓圈和計時器
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None
        if self.temp_circle_timer is not None:
            self.temp_circle_timer.cancel()
            self.temp_circle_timer = None

        # 計算圓圈的寬度和高度（考慮座標軸比例）
        x_range = self.flxs_bound[1] - self.flxs_bound[0]
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

    def remove_temp_circle(self):
        """移除暫時性圓圈"""
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None
            self.fig.canvas.draw_idle()
        self.temp_circle_timer = None


class VisualizeSpet:
    def __init__(
        self, s_spects, s_flxs, s_fpts, flxs, energies, allows, auto_hide=False
    ):
        self.s_spects = s_spects
        self.s_flxs = s_flxs
        self.s_fpts = s_fpts
        self.flxs = flxs
        self.energies = energies
        self.allows = allows
        self.auto_hide = auto_hide  # 新增參數，預設為 False

        # Default scatter point styling
        self.scatter_size = 3
        self.scatter_color = "red"
        self.scatter_color_array = None  # 用於存儲顏色陣列

    def get_flxs(self, mAs, mA_c, period):
        return (mAs - mA_c) / period

    def set_scatter_style(self, size=None, color=None):
        if size is not None:
            self.scatter_size = size
        if color is not None:
            # 可以是顏色名稱或數值陣列
            self.scatter_color = color
            # 檢查是否為陣列
            if isinstance(color, (list, np.ndarray)):
                self.scatter_color_array = color

        return self  # For method chaining

    def create_figure(self):
        fig = go.Figure()

        # Add heatmap traces for each spectrum in s_spects
        for spect in self.s_spects.values():
            # Get corresponding data and range
            data = spect["spectrum"]["data"] ** 1.5
            flx_mask = np.any(~np.isnan(data), axis=0)
            fpt_mask = np.any(~np.isnan(data), axis=1)
            data = data[fpt_mask, :][:, flx_mask]

            # Normalize data
            data = np.abs(data - np.mean(data, axis=0, keepdims=True))
            data /= np.std(data, axis=0, keepdims=True)

            # Add heatmap trace

            sp_flxs = self.get_flxs(
                spect["spectrum"]["mAs"], spect["mA_c"], spect["period"]
            )
            sp_fpts = spect["spectrum"]["fpts"]
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    x=sp_flxs[flx_mask],
                    y=sp_fpts[fpt_mask],
                    colorscale="Greys",
                    showscale=False,
                )
            )

        # Calculate transitions
        fs, labels = energy2transition(self.energies, self.allows)

        # 計算哪些線需要隱藏
        visible_lines = self._filter_nearby_lines(
            fs, self.flxs, self.s_fpts, self.s_flxs
        )

        # Add transition line traces
        for i, label in enumerate(labels):
            visible = "legendonly" if not visible_lines[i] else True
            fig.add_trace(
                go.Scatter(
                    x=self.flxs, y=fs[:, i], mode="lines", name=label, visible=visible
                )
            )

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
                x=self.s_flxs,
                y=self.s_fpts,
                mode="markers",
                marker=marker_dict,
                hovertext=hovertext,
            )
        )

        # Update layout
        fig.update_layout(
            legend_title_text="Transition",
            title_x=0.5,
            xaxis=dict(title="Flux"),
            yaxis=dict(title="Frequency"),
            legend=dict(x=1, y=0.5),
            height=1600,
        )

        sp_flxs = np.concatenate(
            [
                self.get_flxs(s["points"]["mAs"], s["mA_c"], s["period"])
                for s in self.s_spects.values()
            ]
        )
        sp_fpts = np.concatenate([s["points"]["fpts"] for s in self.s_spects.values()])
        flx_bound = (
            min(np.nanmin(sp_flxs), self.s_flxs.min()),
            max(np.nanmax(sp_flxs), self.s_flxs.max()),
        )
        fpt_bound = (
            min(np.nanmin(sp_fpts), self.s_fpts.min()),
            max(np.nanmax(sp_fpts), self.s_fpts.max()),
        )

        # Set x and y axis range
        fig.update_xaxes(range=[flx_bound[0], flx_bound[1]])
        fig.update_yaxes(range=[fpt_bound[0], fpt_bound[1]])

        return fig

    def _filter_nearby_lines(self, fs, flxs, s_fpts, s_flxs):
        """
        計算哪些轉換線靠近散點，並返回布林陣列，決定要顯示哪些線。

        Parameters:
        fs: numpy array, 所有轉換線的頻率數據 (M, K)
        flxs: numpy array, 所有轉換線對應的通量數據 (M, )
        s_fpts: numpy array, 所有散點的頻率數據 (N, )
        s_flxs: numpy array, 所有散點的通量數據 (N, )

        Returns:
        visible_lines: numpy array, 形狀 (K, ), True 表示該線要顯示, False 表示要隱藏
        """
        K = fs.shape[1]

        THRESHOLD = 2

        if self.auto_hide:
            # interpolate flux points
            s_fs = np.array(
                [np.interp(s_flxs, flxs, fs[:, i]) for i in range(fs.shape[1])]
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


def remove_close_points(sp_flxs, sp_fpts, min_fpt_dist=0.005):
    """
    Remove points on the same flux line that are too close to each other.

    Parameters:
    sp_flxs: numpy array, 通量數據 (N, )
    sp_fpts: numpy array, 頻率數據 (N, )
    min_fpt_dist: float, 最小頻率距離

    Returns:
    sp_flxs: numpy array, 去除後的通量數據 (N, )
    sp_fpts: numpy array, 去除後的頻率數據 (N, )
    """
    if len(sp_flxs) == 0:
        return sp_flxs, sp_fpts

    # Sort by frequency first
    sorted_indices = np.argsort(sp_fpts)
    sp_flxs = sp_flxs[sorted_indices]
    sp_fpts = sp_fpts[sorted_indices]

    # stable sort by flux
    sorted_indices = np.argsort(sp_flxs, stable=True)
    sp_flxs = sp_flxs[sorted_indices]
    sp_fpts = sp_fpts[sorted_indices]

    # Remove close points
    prev_i = 0
    mask = np.ones(len(sp_flxs), dtype=bool)
    for i in range(1, len(sp_flxs)):
        if sp_flxs[i] != sp_flxs[prev_i]:
            prev_i = i
            continue

        # Check if the distance is less than the minimum

        if np.abs(sp_fpts[i] - sp_fpts[prev_i]) < min_fpt_dist:
            mask[i] = False
        else:
            prev_i = i

    return sp_flxs[mask], sp_fpts[mask]


def calculate_energy(flxs, EJ, EC, EL, cutoff=50, evals_count=10):
    from scqubits import Fluxonium

    fluxonium = Fluxonium(
        EJ, EC, EL, flux=0.0, cutoff=cutoff, truncated_dim=evals_count
    )
    spectrumData = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    )

    return spectrumData.energy_table


def preprocess_data(flxs, fpts, spectrum):
    fpts = fpts / 1e9  # convert to GHz

    if flxs[0] > flxs[-1]:  # Ensure that the fluxes are in increasing
        flxs = flxs[::-1]
        spectrum = spectrum[:, ::-1]
    if fpts[0] > fpts[-1]:  # Ensure that the frequencies are in increasing
        fpts = fpts[::-1]
        spectrum = spectrum[::-1, :]

    return flxs, fpts, spectrum


def spectrum_analyze(flxs, fpts, signals, threshold, weight=None):
    amps = np.abs(signals - np.ma.mean(signals, axis=0))
    amps /= np.ma.std(amps, axis=0)

    if weight is not None:
        amps *= weight

    s_flxs = []
    s_fpts = []
    for i in range(amps.shape[1]):
        peaks, _ = find_peaks(amps[:, i], height=threshold)
        s_flxs.extend(flxs[i] * np.ones(len(peaks)))
        s_fpts.extend(fpts[peaks])
    return np.array(s_flxs), np.array(s_fpts)


def energy2linearform(energies, allows):
    """
    將能量E轉換為線性形式B,C的躍遷頻率,使得aE的能量對應到|aB+C|的躍遷頻率,其中a可以是任意實數

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    allows: dict, 允許的過渡

    Returns:
    B: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    C: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    names: list, 過渡名稱
    """
    N, M = energies.shape
    K = np.sum([len(v) for v in allows.values() if isinstance(v, list)])
    Bs = np.empty((N, K))
    Cs = np.empty((N, K))
    idx = 0
    for i, j in allows.get("transitions", []):  # E = E_ji
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = 0.0
        idx += 1
    for i, j in allows.get("blue side", []):  # E = E_ji + r_f
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = allows["r_f"]
        idx += 1
    for i, j in allows.get("red side", []):  # E = abs(E_ji - r_f)
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = -allows["r_f"]
        idx += 1
    for i, j in allows.get("mirror", []):  # E = 2 * sample_f - E_ji
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = 2 * allows["sample_f"]
        idx += 1
    for i, j in allows.get("transitions2", []):  # E = 0.5 * E_ji
        Bs[:, idx] = 0.5 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = 0.0
        idx += 1
    for i, j in allows.get("blue side2", []):  # E = 0.5 * E_ji + r_f
        Bs[:, idx] = 0.5 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = allows["r_f"]
        idx += 1
    for i, j in allows.get("red side2", []):  # E = 0.5 * abs(E_ji - r_f)
        Bs[:, idx] = 0.5 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = -0.5 * allows["r_f"]
        idx += 1
    for i, j in allows.get("mirror2", []):  # E = sample_f - 0.5 * E_ji
        Bs[:, idx] = -0.5 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = allows["sample_f"]
        idx += 1

    return Bs, Cs


def energy2transition(energies, allows):
    """
    將能量E轉換為躍遷頻率。

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    allows: dict, 允許的過渡

    Returns:
    fs: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    labels: list, 過渡標籤
    names: list, 過渡名稱
    """
    N, M = energies.shape

    B, C = energy2linearform(energies, allows)
    fs = np.abs(B + C)
    names = []
    for i, j in allows.get("transitions", []):  # E = E_ji
        names.append(f"{i} -> {j}")
    for i, j in allows.get("blue side", []):  # E = E_ji + r_f
        names.append(f"{i} -> {j} blue side")
    for i, j in allows.get("red side", []):  # E = abs(E_ji - r_f)
        names.append(f"{i} -> {j} red side")
    for i, j in allows.get("mirror", []):  # E = 2 * sample_f - E_ji
        names.append(f"{i} -> {j} mirror")
    for i, j in allows.get("transitions2", []):  # E = 0.5 * E_ji
        names.append(f"2 {i} -> {j}")
    for i, j in allows.get("blue side2", []):  # E = 0.5 * E_ji + r_f
        names.append(f"2 {i} -> {j} blue side")
    for i, j in allows.get("red side2", []):  # E = 0.5 * abs(E_ji - r_f)
        names.append(f"2 {i} -> {j} red side")
    for i, j in allows.get("mirror2", []):  # E = sample_f - 0.5 * E_ji
        names.append(f"2 {i} -> {j} mirror")

    return fs, names


@njit(
    "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
    nogil=True,
)
def candidate_breakpoint_search(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, a_min: float, a_max: float
) -> Tuple[float, float]:
    """
    使用候選斷點法尋找最佳的 a 值, 使得目標函數最小化
    目標函數: F(a) = sum_i(min_j(|A[i] - |a * B[i, j] + C[i, j]||))
    假設: A 中所有值都是正的

    Parameters:
    A: 目標向量, numpy 陣列, 形狀 (N,), 所有元素均為正數
    B: 候選向量矩陣, numpy 陣列, 形狀 (N, K)
    C: 偏移矩陣, numpy 陣列, 形狀 (N, K)
    a_min: 最小的 a 值
    a_max: 最大的 a 值

    Returns:
    best_distance: 最小的目標函數值, 如果沒有找到則返回 inf
    best_a: 使得目標函數最小的 a 值, 如果沒有找到則返回 1.0
    """
    N = A.shape[0]
    K = B.shape[1]

    # 找出最佳的 a 值
    best_distance = float("inf")
    best_a = 1.0

    distances = np.empty(N, dtype=np.float64)
    for i in range(N):
        for j in range(K):
            if B[i, j] == 0:
                continue

            a1 = (A[i] - C[i, j]) / B[i, j]
            a2 = (-A[i] - C[i, j]) / B[i, j]

            for a in (a1, a2):
                if a < a_min or a > a_max:
                    continue

                for i in range(N):
                    min_diff = float("inf")
                    for j in range(K):
                        # 計算距離
                        diff = np.abs(A[i] - np.abs(a * B[i, j] + C[i, j]))
                        if diff < min_diff:
                            min_diff = diff
                    distances[i] = min_diff

                total_distance = np.mean(distances)
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_a = a

    return best_distance, best_a


def search_in_database(flxs, fpts, datapath, allows, EJb, ECb, ELb, n_jobs=-1):
    # Load data from database
    with File(datapath, "r") as file:
        f_flxs = file["flxs"][:]  # (f_flxs, )
        f_params = file["params"][:]  # (N, 3)
        f_energies = file["energies"][:]  # (N, f_flxs, M)

    # Interpolate points
    flxs = np.mod(flxs, 1.0)
    sf_energies = np.empty((f_params.shape[0], len(flxs), f_energies.shape[2]))
    for n in range(f_params.shape[0]):
        for m in range(f_energies.shape[2]):
            sf_energies[n, :, m] = np.interp(flxs, f_flxs, f_energies[n, :, m])

    # Initialize variables
    best_idx = 0
    best_factor = 1.0
    best_dist = np.inf
    best_params = np.full(3, np.nan)
    results = np.full((f_params.shape[0], 2), np.nan)  # (N, 2)

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

    # Frequency comparison plot
    ax_freq = fig.add_subplot(gs[:, 0])
    ax_freq.scatter(flxs, fpts, label="Target", color="blue", marker="o")
    pred_scatter = ax_freq.scatter(
        flxs, np.zeros_like(fpts), label="Predicted", color="red", marker="x"
    )

    ax_freq.set_ylabel("Frequency (GHz)")
    ax_freq.legend()
    ax_freq.grid(True)

    # Create scatter plots for EJ, EC,
    param_axs = []
    param_scatters = []
    best_param_scatters = []
    name_bounds = [("EJ", EJb), ("EC", ECb), ("EL", ELb)]
    for i in range(3):
        name, bound = name_bounds[i]
        ax_param = fig.add_subplot(gs[i, 1])
        ax_param.set_xlim(bound[0], bound[1])
        ax_param.set_xlabel(name)
        ax_param.set_ylabel("Distance")
        ax_param.grid()
        scatter = ax_param.scatter(
            range(f_params.shape[0]), np.zeros(f_params.shape[0]), s=2
        )
        best_scatter = ax_param.scatter([0], [0], color="red", s=50, marker="*")
        param_axs.append(ax_param)
        param_scatters.append(scatter)
        best_param_scatters.append(best_scatter)

    dh = display(fig, display_id=True)

    def find_close_points(fpts, energies, factor, allows):
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(factor * Bs + Cs)
        dists = np.abs(fs - fpts[:, None])
        min_idx = np.argmin(dists, axis=1)
        return fs[range(len(fpts)), min_idx]

    prev_draw_idx = -1

    def update_plot(_):
        nonlocal best_dist, best_params, results, prev_draw_idx, best_idx

        # Update best result
        if best_idx != prev_draw_idx:
            p_fpts = find_close_points(fpts, sf_energies[best_idx], best_factor, allows)
            pred_scatter.set_offsets(np.c_[flxs, p_fpts])
            ax_freq.set_ylim(np.min([fpts, p_fpts]), np.max([fpts, p_fpts]))

            fig.suptitle(
                f"Best Distance: {best_dist:.2g}, EJ={best_params[0]:.2f}, EC={best_params[1]:.2f}, EL={best_params[2]:.2f}"
            )
            prev_draw_idx = best_idx

        # Update scatter plots
        dists, factors = results[:, 0], results[:, 1]
        if np.sum(np.isfinite(dists)) > 1:
            for j, (ax, scatter, best_scatter) in enumerate(
                zip(param_axs, param_scatters, best_param_scatters)
            ):
                params_j = f_params[:, j] * factors
                scatter.set_offsets(np.c_[params_j, dists])
                best_scatter.set_offsets(np.c_[best_params[j], best_dist])
                ax.set_ylim(0.0, np.nanmax(dists[np.isfinite(dists)]) * 1.1)

        dh.update(fig)

    def process_energy(i):
        nonlocal f_params, sf_energies, fpts, allows
        param = f_params[i]
        a_min = max(EJb[0] / param[0], ECb[0] / param[1], ELb[0] / param[2])
        a_max = min(EJb[1] / param[0], ECb[1] / param[1], ELb[1] / param[2])
        if a_min > a_max:
            return i, np.inf, 1.0

        Bs, Cs = energy2linearform(sf_energies[i], allows)
        return i, *candidate_breakpoint_search(fpts, Bs, Cs, a_min, a_max)

    idx_bar = trange(f_params.shape[0], desc="Searching...")
    try:
        with AsyncFunc(update_plot) as async_plot:
            for i, dist, factor in Parallel(
                return_as="generator_unordered", n_jobs=n_jobs, require="sharedmem"
            )(delayed(process_energy)(i) for i in idx_bar):
                results[i] = dist, factor

                if not np.isnan(dist) and dist < best_dist:
                    # Update best result
                    best_idx = i
                    best_factor = factor
                    best_params = f_params[i] * factor
                    best_dist = dist

                # Update plot
                async_plot(i)
            else:
                idx_bar.set_description_str("Done! ")
        update_plot(best_idx)

    except KeyboardInterrupt:
        pass
    finally:
        idx_bar.close()
        plt.close(fig)  # Move plt.close(fig) inside finally block

    plt.ion()

    return best_params, fig


def fit_spectrum(flxs, fpts, init_params, allows, param_b, maxfun=1000):
    import scqubits as scq
    from scipy.optimize import minimize

    scq.settings.PROGRESSBAR_DISABLED, old = True, scq.settings.PROGRESSBAR_DISABLED

    evals_count = 0
    for lvl in allows.values():
        if not isinstance(lvl, list) or len(lvl) == 0:
            continue
        evals_count = max(evals_count, *[max(lv) for lv in lvl])
    evals_count += 1

    fluxonium = scq.Fluxonium(
        *init_params, flux=0.0, truncated_dim=evals_count, cutoff=40
    )

    pbar = tqdm(
        desc=f"({init_params[0]:.2f}, {init_params[1]:.2f}, {init_params[2]:.2f})",
        total=maxfun,
    )

    def callback(intermediate_result):
        pbar.update(1)
        if isinstance(intermediate_result, np.ndarray):
            # old version
            cur_params = intermediate_result
        else:
            cur_params = intermediate_result.x
        pbar.set_description(
            f"({cur_params[0]:.4f}, {cur_params[1]:.4f}, {cur_params[2]:.4f})"
        )

    def params2energy(flxs, params):
        nonlocal fluxonium, evals_count

        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]
        return fluxonium.get_spectrum_vs_paramvals(
            "flux", flxs, evals_count=evals_count, get_eigenstates=True
        ).energy_table

    # Find unique values in flxs and map original indices to unique indices
    uni_flxs, uni_idxs = np.unique(flxs, return_inverse=True)

    def loss_func(param):
        nonlocal fluxonium, flxs, uni_flxs, uni_idxs, allows, fpts

        energies = params2energy(uni_flxs, param)
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(Bs + Cs)[uni_idxs, :]
        return np.sum(np.sqrt(np.min(np.abs(fpts[:, None] - fs), axis=1)))

    res = minimize(
        loss_func,
        init_params,
        bounds=param_b,
        method="L-BFGS-B",
        options={"maxfun": maxfun},
        callback=callback,
    )

    pbar.close()

    scq.settings.PROGRESSBAR_DISABLED = old

    if isinstance(res, np.ndarray):  # old version
        best_params = res
    else:
        best_params = res.x

    return best_params


def dump_result(path, name, params, cflx, period, allows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {
                "name": name,
                "params": {
                    "EJ": params[0],
                    "EC": params[1],
                    "EL": params[2],
                },
                "half flux": cflx,
                "period": period,
                "allows": allows,
            },
            f,
            indent=4,
        )


def load_result(path):
    with open(path, "r") as f:
        data = json.load(f)

    return (
        data["name"],
        np.array([data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]),
        data["half flux"],
        data["period"],
        data["allows"],
    )
