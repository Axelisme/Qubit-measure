from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.animation import FuncAnimation

from ..processing import cast2real_and_norm, diff_mirror


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
        spect_center = (mAs[0] + mAs[-1]) / 2
        self.mA_c = spect_center if mA_c is None else mA_c
        self.mA_e = mAs[-5] if mA_e is None else mA_e
        if mA_c is not None and mA_e is not None:
            period = 2 * abs(self.mA_e - self.mA_c)

            # fold the mA_c and mA_e to the closest point to the spect_center
            self.mA_c = (
                self.mA_c - round((self.mA_c - spect_center) / period, 0) * period
            )
            self.mA_e = (
                self.mA_e - round((self.mA_e - spect_center) / period, 0) * period
            )
        self.mA_c = float(self.mA_c)
        self.mA_e = float(self.mA_e)

        self.mAs = mAs
        self.fpts = fpts
        self.spectrum = spectrum

        # 僅使用振幅 (magnitude) 模式，預設為 False
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
        self.min_dist = 0.1 * abs(mAs[-1] - mAs[0])
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
            interval=500,
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
        total_width = abs(self.mAs[-1] - self.mAs[0])
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
        precision = 0.25 * (self.mAs.max() - self.mAs.min()) / len(self.mAs)

        # 定義搜索範圍
        left_bound = max(self.mAs.min(), current_pos - search_width / 2)
        right_bound = min(self.mAs.max(), current_pos + search_width / 2)

        # 創建候選位置，只考慮半格的整數倍
        # 將邊界轉換為相對於起始點的格數
        left_steps = int(np.floor((left_bound - self.mAs.min()) / precision))
        right_steps = int(np.ceil((right_bound - self.mAs.min()) / precision))

        # 生成候選位置
        candidates = [
            self.mAs.min() + i * precision for i in range(left_steps, right_steps + 1)
        ]

        # 確保候選位置在有效範圍內
        candidates = [
            pos for pos in candidates if self.mAs.min() <= pos <= self.mAs.max()
        ]

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
        elif blue_dist <= red_dist and blue_dist < self.min_dist / 2:
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
        self.zoom_im.autoscale()

        mirror_loss = np.mean(diff_amps[diff_amps != 0.0])

        # set axis limits to simulate zoom
        Dx = 0.3 * abs(self.mAs[-1] - self.mAs[0])
        Dy = 0.3 * abs(self.fpts[-1] - self.fpts[0])
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

    def get_positions(self, finish: bool = True) -> Tuple[float, float]:
        """運行交互式選擇器並返回兩條線的位置"""
        if not self.is_finished and finish:
            self.finish_interactive()
        precision = 0.5 * (self.mAs[-1] - self.mAs[0]) / len(self.mAs)
        mA_c = precision * round((self.mA_c - self.mAs[0]) / precision) + self.mAs[0]
        mA_e = precision * round((self.mA_e - self.mAs[0]) / precision) + self.mAs[0]
        return float(mA_c), float(mA_e)

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
