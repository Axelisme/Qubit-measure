from typing import Any, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class InteractiveOneTone:
    def __init__(
        self,
        mAs: np.ndarray,
        fpts: np.ndarray,
        spectrum: np.ndarray,
        threshold: float = 1.0,
    ) -> None:
        self.mAs = mAs
        self.fpts = fpts
        self.spectrum = spectrum
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
        self.amps2d = np.abs(self.spectrum)  # (fpts, mAs)

        self.max_idx = np.argmax(np.mean(self.amps2d, axis=1))
        self.img = self.axes[0].imshow(
            self.amps2d,
            aspect="auto",
            origin="lower",
            extent=(self.mAs[0], self.mAs[-1], self.fpts[0], self.fpts[-1]),
        )
        self.line = self.axes[0].axhline(
            self.fpts[self.max_idx],
            color="red",
            label="max fpts",
        )

        # 顯示1D切面
        self.amps = self.amps2d[self.max_idx, :]  # (mAs,)

        self.smoothed_amps = gaussian_filter1d(
            np.max(self.amps) - self.amps, sigma=1
        )  # smooth the signal
        self.smoothed_amps /= np.std(self.smoothed_amps)  # normalize the signal

        (self.curve,) = self.axes[1].plot(self.mAs, self.smoothed_amps)

        # 找峰值並顯示
        self.update_peaks(threshold)

        # 設置軸標籤
        self.axes[0].set_ylabel("Frequency (GHz)")
        self.axes[1].set_xlabel("Current (mA)")
        self.axes[1].set_ylabel("Normalized Amplitude")

    def update_peaks(self, threshold: float) -> None:
        """更新峰值點"""

        # 找出峰值
        peaks, _ = find_peaks(self.smoothed_amps, prominence=threshold)

        # 獲取對應的 mAs 和 fpts
        self.s_mAs = self.mAs[peaks]
        self.s_fpts = np.full_like(self.s_mAs, self.fpts[self.max_idx])

        # 更新圖表上的峰值標記
        if hasattr(self, "scatter1"):
            self.scatter1.remove()
        if hasattr(self, "scatter2"):
            self.scatter2.remove()

        # 在上圖中標記所選點
        self.scatter1 = self.axes[0].scatter(
            self.s_mAs, self.s_fpts, color="red", s=30, zorder=5
        )

        # 在下圖中標記峰值
        self.scatter2 = self.axes[1].scatter(
            self.s_mAs, self.smoothed_amps[peaks], color="red", s=30, zorder=5
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

    def get_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回找到的點位置"""
        if not self.is_finished:
            self.on_finish(None)
        return self.s_mAs, self.s_fpts
