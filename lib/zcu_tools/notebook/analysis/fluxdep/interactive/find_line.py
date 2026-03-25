from __future__ import annotations

import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import Optional

from ..processing import cast2real_and_norm, diff_mirror


class InteractiveLines:
    TRACK_INFO = {
        "half flux": "<span style='color:red'>正在移動half flux(紅線)</span>",
        "integer flux": "<span style='color:blue'>正在移動integer flux(藍線)</span>",
        "none": "<span style='color:gray'>未選擇</span>",
    }

    def __init__(
        self,
        signals: NDArray,
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
    ) -> None:
        plt.ioff()  # 避免立即顯示圖表
        self.fig_main, self.ax_main = plt.subplots(figsize=(4, 3))
        self.fig_loss, self.ax_loss = plt.subplots(figsize=(4, 3))
        self.fig_main.tight_layout()
        self.fig_loss.tight_layout()
        plt.ion()

        # 初始化線的位置
        flux_center = (dev_values[0] + dev_values[-1]) / 2
        self.flux_half = flux_center if flux_half is None else flux_half
        self.flux_int = dev_values[-5] if flux_int is None else flux_int
        if flux_half is not None and flux_int is not None:
            fix_period = 2 * abs(self.flux_int - self.flux_half)

            # fold the flux_half and flux_int to the closest point to the spect_center
            self.flux_half = (
                self.flux_half
                - round((self.flux_half - flux_center) / fix_period, 0) * fix_period
            )
            self.flux_int = (
                self.flux_int
                - round((self.flux_int - flux_center) / fix_period, 0) * fix_period
            )
        self.flux_half = float(self.flux_half)
        self.flux_int = float(self.flux_int)

        self.dev_values = dev_values
        self.freqs = freqs
        self.signals = signals

        # 僅使用振幅 (magnitude) 模式，預設為 False
        self.only_use_magnitude: bool = False

        # 根據 only_use_magnitude 計算顯示用資料
        self.real_signals = cast2real_and_norm(
            signals, use_phase=not self.only_use_magnitude
        )

        self.mouse_x = None
        self.mouse_y = None
        self.prev_mouse_x = None
        self.prev_mouse_y = None

        # Flag to確認滑鼠是否真的移動過，用於減少 update_zoom_view 次數
        self._mouse_moved = False

        self.create_widgets()
        self.create_background(dev_values, freqs, self.real_signals)
        self.create_lines(dev_values)
        self.create_loss(dev_values, freqs, self.real_signals)

        # 顯示 widget
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
                    widgets.HBox([self.fig_main.canvas, self.fig_loss.canvas]),
                ]
            )
        )

    def create_widgets(self) -> None:
        """創建 ipywidgets 控件"""
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

        # 新增: 僅使用振幅顯示的切換開關
        self.only_use_magnitude_checkbox = widgets.Checkbox(
            value=self.only_use_magnitude, description="Magnitude Only"
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
        self.status_text = widgets.HTML(value="<span style='color:gray'>未選擇</span>")

        # 綁定事件
        self.flux_half_button.on_click(self.set_picked_half_flux)
        self.flux_int_button.on_click(self.set_picked_int_flux)
        self.finish_button.on_click(self.on_finish)
        self.swap_button.on_click(self.swap_lines)
        self.auto_align_button.on_click(self.auto_align_lines)

        # 綁定 magnitude 顯示模式切換
        self.only_use_magnitude_checkbox.observe(
            self.on_toggle_magnitude, names="value"
        )

    def create_background(self, dev_values, freqs, real_signals) -> None:
        """創建背景圖片"""
        # 儲存圖像控制柄以便之後更新資料
        dx = (dev_values[-1] - dev_values[0]) / (len(dev_values) - 1)
        dy = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
        self.main_im = self.ax_main.imshow(
            real_signals.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(
                dev_values[0] - dx / 2,
                dev_values[-1] + dx / 2,
                freqs[0] - dy / 2,
                freqs[-1] + dy / 2,
            ),
        )

        self.ax_main.set_xlim(dev_values[0], dev_values[-1])
        self.ax_main.set_ylim(freqs[0], freqs[-1])

    def create_lines(self, dev_values) -> None:
        """創建兩條垂直線"""
        # 創建兩條垂直線
        self.half_line = self.ax_main.axvline(
            x=self.flux_half, color="r", linestyle="--"
        )
        self.int_line = self.ax_main.axvline(x=self.flux_int, color="b", linestyle="--")

        # 設置變數
        self.picked = None
        self.min_flux_dist = 0.01 * abs(dev_values[-1] - dev_values[0])
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

    def create_loss(self, dev_values, freqs, real_signals) -> None:
        """創建mirror loss視圖"""
        self.ax_loss.set_title(f"mirror loss: {None}")

        dx = (dev_values[-1] - dev_values[0]) / (len(dev_values) - 1)
        dy = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
        self.loss_im = self.ax_loss.imshow(
            real_signals.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(
                dev_values[0] - dx / 2,
                dev_values[-1] + dx / 2,
                freqs[0] - dy / 2,
                freqs[-1] + dy / 2,
            ),
        )
        self.ax_loss.set_xticks([])
        self.ax_loss.set_yticks([])

        # show red spot at center
        x = self.flux_half
        y = 0.5 * (freqs[0] + freqs[-1])
        self.loss_dot = self.ax_loss.plot([x], [y], "ro")[0]

        self.anim_loss = FuncAnimation(
            self.fig_loss,
            self.update_loss_view,
            interval=500,
            blit=True,
            cache_frame_data=False,
        )

    def get_info(self) -> str:
        return f"half flux: {self.flux_half:.2e}, integer flux: {self.flux_int:.2e}, flux period: {2 * abs(self.flux_int - self.flux_half):.2e}"

    def set_picked_half_flux(self, _) -> None:
        """選擇half flux(紅線)"""
        if self.is_finished:
            return

        if self.active_line == self.half_line:
            # 如果已經在移動紅線, 則停止移動
            self.stop_tracking()
        else:
            # 開始移動紅線
            self.active_line = self.half_line
            self.picked = self.half_line
            self.status_text.value = self.TRACK_INFO["half flux"]

    def set_picked_int_flux(self, _) -> None:
        """選擇integer flux(藍線)"""
        if self.is_finished:
            return

        if self.active_line == self.int_line:
            # 如果已經在移動藍線, 則停止移動
            self.stop_tracking()
        else:
            # 開始移動藍線
            self.active_line = self.int_line
            self.picked = self.int_line
            self.status_text.value = self.TRACK_INFO["integer flux"]

    def stop_tracking(self) -> None:
        """停止追蹤滑鼠"""
        self.active_line = None
        self.picked = None
        self.status_text.value = self.TRACK_INFO["none"]

    def swap_lines(self, _) -> None:
        """交換half flux線與integer flux線的位置"""
        if self.is_finished:
            return

        # 停止當前的追蹤
        self.stop_tracking()

        # 交換線條的位置
        self.flux_half, self.flux_int = self.flux_int, self.flux_half

        # 更新線條的視覺位置
        self.half_line.set_xdata([self.flux_half])
        self.int_line.set_xdata([self.flux_int])

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
        total_width = abs(self.dev_values[-1] - self.dev_values[0])
        search_width = total_width / 20

        # 為紅線和藍線分別找到最佳位置
        best_flux_half = self._find_best_position(self.flux_half, search_width)
        best_flux_int = self._find_best_position(self.flux_int, search_width)

        # 更新線條位置
        self.flux_half = best_flux_half
        self.flux_int = best_flux_int

        # 更新線條的視覺位置
        self.half_line.set_xdata([self.flux_half])
        self.int_line.set_xdata([self.flux_int])

        # 更新位置文字
        self.position_text.value = self.get_info()

        # 重新繪製
        self.fig_main.canvas.draw_idle()

    def _find_best_position(self, current_pos: float, search_width: float) -> float:
        """在給定範圍內找到mirror loss最小的位置"""
        # 計算半格精度
        precision = (
            0.25
            * (self.dev_values.max() - self.dev_values.min())
            / len(self.dev_values)
        )

        # 定義搜索範圍
        left_bound = max(self.dev_values.min(), current_pos - search_width / 2)
        right_bound = min(self.dev_values.max(), current_pos + search_width / 2)

        # 創建候選位置，只考慮半格的整數倍
        # 將邊界轉換為相對於起始點的格數
        left_steps = int(np.floor((left_bound - self.dev_values.min()) / precision))
        right_steps = int(np.ceil((right_bound - self.dev_values.min()) / precision))

        # 生成候選位置
        candidates = [
            self.dev_values.min() + i * precision
            for i in range(left_steps, right_steps + 1)
        ]

        # 確保候選位置在有效範圍內
        candidates = [
            pos
            for pos in candidates
            if self.dev_values.min() <= pos <= self.dev_values.max()
        ]

        # 如果沒有有效的候選位置，返回當前位置
        if not candidates:
            return current_pos

        best_pos = current_pos
        min_loss = float("inf")

        real_signals = cast2real_and_norm(
            self.signals, use_phase=not self.only_use_magnitude
        )

        # 對每個候選位置計算mirror loss
        pbar = None
        start_t = time.time()
        for i, candidate in enumerate(candidates):
            # 計算該位置的mirror loss
            # 總是使用spectrum來計算(包含phase資訊)
            diff_amps = diff_mirror(self.dev_values, real_signals, candidate)
            valid_amps = diff_amps[diff_amps != 0.0]

            # 確保有有效的數據點
            if len(valid_amps) > 0:
                mirror_loss = np.mean(valid_amps)

                # 如果這個位置的loss更小，更新最佳位置
                if not np.isnan(mirror_loss) and mirror_loss < min_loss:
                    min_loss = mirror_loss
                    best_pos = candidate

            if pbar is None:
                if time.time() - start_t > 0.5:  # 如果計算超過0.5秒，才顯示進度條
                    pbar = tqdm(
                        candidates, desc="Auto-aligning", unit="pos", leave=False
                    )
                    pbar.update(i + 1)
            else:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        return best_pos

    def onclick(self, event) -> None:
        """滑鼠點擊事件"""
        if self.is_finished or event.inaxes != self.ax_main:
            return

        flux_half_dist = abs(event.xdata - self.half_line.get_xdata()[0])
        flux_int_dist = abs(event.xdata - self.int_line.get_xdata()[0])

        # 如果已經有活動的線條, 點擊任何位置都停止追蹤
        if self.active_line is not None:
            self.stop_tracking()
            return

        # 選擇最近的線
        if flux_half_dist < flux_int_dist and flux_half_dist < 3 * self.min_flux_dist:
            self.set_picked_half_flux(None)
        elif flux_int_dist <= flux_half_dist and flux_int_dist < 3 * self.min_flux_dist:
            self.set_picked_int_flux(None)

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

        other_line = self.int_line if self.picked is self.half_line else self.half_line
        other_x = other_line.get_xdata()[0]

        # 確保線之間保持最小距離
        if x > other_x and x - other_x < self.min_flux_dist:
            x = other_x + self.min_flux_dist
        elif x < other_x and other_x - x < self.min_flux_dist:
            x = other_x - self.min_flux_dist

        # 更新線的位置
        if self.conjugate_checkbox.value:
            # 同步移動
            dx = x - self.picked.get_xdata()[0]
            # 更新兩條線的位置
            self.half_line.set_xdata([self.half_line.get_xdata()[0] + dx])
            self.int_line.set_xdata([self.int_line.get_xdata()[0] + dx])
        else:
            # 單獨移動
            self.picked.set_xdata([x])

        # 更新位置文字
        self.flux_half = self.half_line.get_xdata()[0]
        self.flux_int = self.int_line.get_xdata()[0]
        self.position_text.value = self.get_info()

        return [self.half_line, self.int_line]

    def update_loss_view(self, _) -> list:
        """更新放大視圖"""
        x, y = self.mouse_x, self.mouse_y

        # 條件不足或滑鼠未移動時不更新，減少 CPU loading
        if x is None or y is None or self.active_line is None or not self._mouse_moved:
            return []  # do nothing

        # reset flag
        self._mouse_moved = False

        # 總是使用spectrum來計算(包含phase資訊)
        mirror_loss = diff_mirror(self.dev_values, self.signals, x)
        self.loss_im.set_data(mirror_loss.T)
        self.loss_im.autoscale()

        mirror_loss = np.mean(mirror_loss[mirror_loss != 0.0])

        # set axis limits to simulate zoom
        Dx = 0.3 * abs(self.dev_values[-1] - self.dev_values[0])
        Dy = 0.3 * abs(self.freqs[-1] - self.freqs[0])
        self.ax_loss.set_xlim(x - Dx, x + Dx)
        self.ax_loss.set_ylim(y - Dy, y + Dy)
        self.ax_loss.set_title(f"mirror loss: {mirror_loss:.4f}")

        self.loss_dot.set_xdata([x])
        self.loss_dot.set_ydata([y])
        self.loss_dot.set_color("r" if self.active_line is self.half_line else "b")

        return [self.loss_im, self.loss_dot]

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
        self.anim_loss.event_source.stop()
        plt.close(self.fig_main)
        plt.close(self.fig_loss)

    def get_positions(self, finish: bool = True) -> tuple[float, float]:
        """運行交互式選擇器並返回兩條線的位置"""
        if not self.is_finished and finish:
            self.finish_interactive()
        return float(self.flux_half), float(self.flux_int)

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
            self.signals, use_phase=not self.only_use_magnitude
        )

        self.main_im.set_data(self.real_signals.T)
        self.main_im.autoscale()

        # 更新背景影像
        self.fig_main.canvas.draw_idle()
