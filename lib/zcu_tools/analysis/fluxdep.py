import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from tqdm.auto import tqdm


class InteractiveSelector:
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
        self.s_flxs, self.s_fpts, self.s_ids = spectrum_analyze(
            flxs, fpts, spectrum, threshold, weight=self.mask
        )
        self.scatter = self.ax.scatter(self.s_flxs, self.s_fpts, color="r", s=2)

    def init_callback(self):
        # 綁定事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)

    def update_points(self):
        threshold = self.threshold_slider.value
        self.s_flxs, self.s_fpts, _ = spectrum_analyze(
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
    s_ids = []
    for i in range(amps.shape[1]):
        peaks, _ = find_peaks(amps[:, i], height=threshold)
        s_flxs.extend(flxs[i] * np.ones(len(peaks)))
        s_fpts.extend(fpts[peaks])
        s_ids.extend([(i, j) for j in peaks])
    return np.array(s_flxs), np.array(s_fpts), np.array(s_ids)


def remove_close_points(flxs, fpts, dist_ratio):
    # remove some close points
    if len(flxs) < 2:
        return flxs, fpts  # ignore in edge case

    mask = np.ones(len(flxs), dtype=bool)
    t_d2 = np.sqrt((flxs[-1] - flxs[0]) ** 2 + (fpts[-1] - fpts[0]) ** 2) * dist_ratio
    prev = 0
    for i in range(1, len(flxs)):
        d_flx = flxs[i] - flxs[prev]
        d_fs = fpts[i] - fpts[prev]
        d2 = np.sqrt(d_flx**2 + d_fs**2)
        if d2 < t_d2:
            mask[i] = False
        else:
            prev = i

    return flxs[mask], fpts[mask]


def energy2transition(energies, allows):
    # energies: shape (n, m)
    fs = []
    labels = []
    names = []
    for i, j in allows.get("transitions", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(freq)
        labels.append(((i, -np.ones_like(freq)), (j, np.ones_like(freq))))
        names.append(f"{i} -> {j}")
    for i, j in allows.get("blue side", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(allows["r_f"] + freq)
        labels.append(((i, -np.ones_like(freq)), (j, np.ones_like(freq))))
        names.append(f"{i} -> {j} blue side")
    for i, j in allows.get("red side", []):
        freq = energies[:, j] - energies[:, i]
        red_f = allows["r_f"] - freq
        fs.append(np.abs(red_f))
        mask = np.where(red_f > 0, 1, -1)
        # labels.append([(i, mask), (j, -mask)])
        labels.append(((i, mask), (j, -mask)))
        names.append(f"{i} -> {j} red side")
    for i, j in allows.get("mirror", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(2 * allows["sample_f"] - freq)
        labels.append(((i, np.ones_like(freq)), (j, -np.ones_like(freq))))
        names.append(f"{i} -> {j} mirror")
    for i, j in allows.get("transitions2", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(0.5 * freq)
        labels.append(((i, -0.5 * np.ones_like(freq)), (j, 0.5 * np.ones_like(freq))))
        names.append(f"{i} -> {j} 2")
    for i, j in allows.get("blue side2", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(0.5 * (allows["r_f"] + freq))
        labels.append(((i, -0.5 * np.ones_like(freq)), (j, 0.5 * np.ones_like(freq))))
        names.append(f"{i} -> {j} blue side 2")
    for i, j in allows.get("red side2", []):
        freq = energies[:, j] - energies[:, i]
        red_f = allows["r_f"] - freq
        fs.append(0.5 * np.abs(red_f))
        mask = np.where(red_f > 0, 1, -1)
        labels.append(((i, 0.5 * np.ones_like(freq)), (j, -0.5 * np.ones_like(freq))))
        names.append(f"{i} -> {j} red side 2")
    for i, j in allows.get("mirror2", []):
        freq = energies[:, j] - energies[:, i]
        fs.append(allows["sample_f"] - 0.5 * freq)
        labels.append(((i, 0.5 * np.ones_like(freq)), (j, -0.5 * np.ones_like(freq))))
        names.append(f"{i} -> {j} mirror 2")

    return np.array(fs).T, labels, names


def candidate_breakpoint_search(A, B):
    """
    使用候選斷點法尋找最佳的 a 值, 使得目標函數最小化
    目標函數: F(a) = sum_i(min_j(|A[i] - a * B[j, i]|))

    輸入:
      A: 目標向量, numpy 陣列, 形狀 (N,)
      B: 候選向量矩陣, numpy 陣列, 形狀 (M, N), 其中每一列代表一個候選向量

    輸出:
      best_distance: 最小的目標函數值
      best_a: 使得目標函數最小的 a 值
    """
    N = A.shape[0]
    M = B.shape[0]

    if B.shape[1] != N:
        raise ValueError(
            f"A and B must have the same number of columns, but got A={A.shape} and B={B.shape}"
        )

    # 收集所有候選的 a 值
    candidate_as = []
    for i in range(N):
        for j in range(M):
            # 若 B[j, i] 為 0 則略過
            if B[j, i] != 0:
                a_candidate = A[i] / B[j, i]
                # 只考慮正的 a
                if a_candidate > 0:
                    candidate_as.append(a_candidate)

    # 若沒有候選點, 則返回 None
    if len(candidate_as) == 0:
        return None, None

    # 去除重複並排序
    candidate_as = np.unique(np.array(candidate_as))  # shape (K,)

    # 對每個候選 a 評估目標函數
    # use broadcasting to calculate a * B, shape (K, M, N)
    aB = candidate_as[:, None, None] * B[None, ...]
    # calculate the minimum difference for each i
    min_diff = np.min(np.abs(aB - A), axis=1)  # shape (K, N)
    total_distance = np.sum(min_diff, axis=1)  # shape (K,)

    best_idx = np.argmin(total_distance)
    return total_distance[best_idx], candidate_as[best_idx]


def search_in_database(flxs, fpts, datapath, allows, n_jobs=-1):
    from h5py import File
    from joblib import Parallel, delayed

    with File(datapath, "r") as file:
        h_flxs = file["flxs"][:]
        h_params = file["params"][:]
        h_energies = file["energies"][:]

    def dist_to_curve(energies):
        fs, *_ = energy2transition(energies, allows)
        return candidate_breakpoint_search(fpts, fs.T)  # (distance, a)

    # 找到最接近的 energy index
    flxs = np.mod(flxs, 1.0)
    d2 = (h_flxs[:, None] - flxs[None, :]) ** 2
    idxs = np.argmin(d2, axis=0)

    # 平行計算距離
    def process_energy(i):
        energy = h_energies[i, idxs]
        dist, a = dist_to_curve(energy)
        if dist is not None:
            return dist, h_energies[i], h_params[i] * a
        return float("inf"), None, None

    # compile candidate_breakpoint_search
    dist_to_curve(h_energies[0, idxs])

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_energy)(i)
        for i in tqdm(range(h_params.shape[0]), desc="Searching...")
    )

    # 找到最佳結果
    _, best_energy, best_params = min(results, key=lambda x: x[0])

    return best_params, h_flxs, best_energy


def fit_spectrum(flxs, fpts, init_params, allows, params_b=None, maxfun=1000):
    import scqubits as scq
    from scipy.optimize import minimize

    scq.settings.PROGRESSBAR_DISABLED, old = True, scq.settings.PROGRESSBAR_DISABLED

    max_level = 0
    for lvl in allows.values():
        if not isinstance(lvl, list):
            continue
        max_level = max(max_level, *[max(lv) for lv in lvl])
    max_level += 1
    fluxonium = scq.Fluxonium(
        *init_params, flux=0.0, truncated_dim=max_level, cutoff=50
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
            f"({cur_params[0]:.2f}, {cur_params[1]:.2f}, {cur_params[2]:.2f})"
        )

    def params2spec(fluxonium, flxs, params):
        nonlocal max_level

        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]
        return fluxonium.get_spectrum_vs_paramvals(
            "flux", flxs, evals_count=max_level, get_eigenstates=True
        )

    def energies2loss(energies, fpts, allows):
        # energies: (n, l)
        fs, labels, _ = energy2transition(energies, allows)
        dist = fs - fpts[:, None]  # (n, m)
        dist2 = dist**2

        idxs = np.arange(len(energies))
        min_idx = np.argmin(dist2, axis=1)  # (n, )
        grad_energies = np.zeros_like(energies)
        for i, min_id in enumerate(min_idx):
            f, t = labels[min_id]
            grad_energies[i, f[0]] += 2 * dist[i, min_idx[i]] * f[1][i]
            grad_energies[i, t[0]] += 2 * dist[i, min_idx[i]] * t[1][i]

        return np.sum(dist2[idxs, min_idx]), grad_energies

    def get_dH_dE(fluxonium, params, spect):
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]
        # H = 4Ec*n^2 - EJ*cos(phi) + 0.5*EL phi^2
        dH_dE = np.empty((*spect.energy_table.shape, 3))
        for i in range(len(spect.energy_table)):
            flx = spect.param_vals[i]
            eval, estate = spect.energy_table[i], spect.state_table[i]
            cosphi_op = fluxonium.cos_phi_operator(1, 2 * np.pi * flx, (eval, estate))
            n_op = fluxonium.n_operator((eval, estate))
            phi_op = fluxonium.phi_operator((eval, estate))
            dH_dE[i, :, 0] = -np.diag(cosphi_op).real
            dH_dE[i, :, 1] = 4 * np.sum(n_op * n_op.T, axis=1).real
            dH_dE[i, :, 2] = 0.5 * np.sum(phi_op * phi_op.T, axis=1).real
        return dH_dE

    def loss_func(params):
        nonlocal fluxonium, flxs, allows, fpts

        spectrumData = params2spec(fluxonium, flxs, params)
        loss, grad = energies2loss(spectrumData.energy_table, fpts, allows)

        dH_dE = get_dH_dE(fluxonium, params, spectrumData)  # (n, l, 3)

        grad_params = np.sum(grad[:, :, None] * dH_dE, axis=(0, 1))

        return loss, grad_params

    res = minimize(
        loss_func,
        init_params,
        bounds=params_b,
        method="L-BFGS-B",
        options={"maxfun": maxfun},
        jac=True,
        callback=callback,
    )

    pbar.close()

    scq.settings.PROGRESSBAR_DISABLED = old

    if isinstance(res, np.ndarray):  # old version
        return res
    return res.x
