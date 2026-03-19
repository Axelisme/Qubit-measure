from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing_extensions import Any


class InteractiveOneTone:
    def __init__(
        self,
        signals: NDArray[np.complex128],
        dev_values: NDArray[np.float64],
        freqs: NDArray[np.float64],
        threshold: float = 1.0,
    ) -> None:
        self.signals = signals
        self.dev_values = dev_values
        self.freqs = freqs
        self.is_finished = False

        plt.ioff()  # to avoid showing the plot immediately
        self.fig, self.axes = plt.subplots(2, 1, figsize=(8, 5))
        self.fig.tight_layout()
        plt.ion()

        # 顯示頻譜
        self.init_plots(threshold)

        # 創建控制元件
        self.threshold_slider = widgets.FloatSlider(
            value=threshold, min=0.0, max=3.0, step=0.01, description="Threshold:"
        )
        self.threshold_slider.observe(self.on_threshold_change, names="value")

        self.finish_button = widgets.Button(
            description="Finish", button_style="success"
        )
        self.finish_button.on_click(self.on_finish)

        # 顯示 widget
        display(
            widgets.HBox(
                [
                    self.fig.canvas,
                    widgets.VBox(
                        [
                            self.threshold_slider,
                            self.finish_button,
                        ]
                    ),
                ]
            )
        )

    def init_plots(self, threshold: float) -> None:
        """初始化圖表"""
        # 顯示2D頻譜
        self.real_signals = np.abs(self.signals)  # (mAs, fpts)

        abs_grad = (
            np.abs(self.signals[:, 1:] - self.signals[:, :-1])
            / (self.freqs[1:] - self.freqs[:-1])[None]
        )
        rel_grad = abs_grad / (
            np.clip(np.abs(self.signals[:, 1:] + self.signals[:, :-1]), 1e-12, None)
        )
        rel_grad = gaussian_filter1d(rel_grad, sigma=1, axis=1)

        self.max_freq_idx = np.argmax(np.mean(rel_grad, axis=0))

        self.img = self.axes[0].imshow(
            self.real_signals.T,
            aspect="auto",
            origin="lower",
            extent=(
                self.dev_values[0],
                self.dev_values[-1],
                self.freqs[0],
                self.freqs[-1],
            ),
        )
        self.line = self.axes[0].axhline(
            self.freqs[self.max_freq_idx],
            color="red",
            label="max fpts",
        )

        # 顯示1D切面
        self.real_signals_slice = self.real_signals[:, self.max_freq_idx]  # (mAs,)

        self.smoothed_real_signals = gaussian_filter1d(
            np.max(self.real_signals_slice) - self.real_signals_slice, sigma=1
        )
        self.smoothed_real_signals /= np.std(self.smoothed_real_signals)

        (self.curve,) = self.axes[1].plot(self.dev_values, self.smoothed_real_signals)
        self.axes[1].set_xlim(self.dev_values[0], self.dev_values[-1])

        # 找峰值並顯示
        self.update_peaks(threshold)

        # 設置軸標籤
        self.axes[0].set_ylabel("Frequency (GHz)")
        self.axes[1].set_xlabel("Current (mA)")
        self.axes[1].set_ylabel("Normalized Amplitude")

    def update_peaks(self, threshold: float) -> None:
        """更新峰值點"""

        # 找出峰值
        peaks, _ = find_peaks(self.smoothed_real_signals, prominence=threshold)

        # 獲取對應的 mAs 和 fpts
        self.s_dev_values = self.dev_values[peaks]
        self.s_freqs = np.full_like(self.s_dev_values, self.freqs[self.max_freq_idx])

        # 更新圖表上的峰值標記
        if hasattr(self, "scatter1"):
            self.scatter1.remove()
        if hasattr(self, "scatter2"):
            self.scatter2.remove()

        # 在上圖中標記所選點
        self.scatter1 = self.axes[0].scatter(
            self.s_dev_values, self.s_freqs, color="red", s=30, zorder=5
        )

        # 在下圖中標記峰值
        self.scatter2 = self.axes[1].scatter(
            self.s_dev_values,
            self.smoothed_real_signals[peaks],
            color="red",
            s=30,
            zorder=5,
        )

        # 更新圖表
        self.fig.canvas.draw_idle()

    def on_threshold_change(self, change: Any) -> None:
        """當閾值變更時更新圖表"""
        if self.is_finished:
            return

        self.update_peaks(change.new)

    def on_finish(self, _: Any) -> None:
        """完成按鈕的回調函數"""
        plt.close(self.fig)
        self.is_finished = True

        # also clear the output
        clear_output(wait=False)

    def get_positions(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """返回找到的點位置"""
        if not self.is_finished:
            self.on_finish(None)
        return self.s_dev_values, self.s_freqs
