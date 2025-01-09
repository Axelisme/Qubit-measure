import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange


class InteractiveSelector:
    def __init__(self, spectrum, flxs, fpts, s_flxs, s_fpts, colors=None):
        from matplotlib.animation import FuncAnimation

        self.fig, self.ax = plt.subplots()
        self.colors = (
            np.array(["r" for _ in range(len(s_flxs))]) if colors is None else colors
        )
        self.current_color = "r"
        self.is_selecting = False
        self.mouse_x = None
        self.mouse_y = None
        self.is_dragging = False  # 新增：追蹤滑鼠按住狀態

        # 顯示光譜圖
        self.ax.imshow(
            spectrum,
            aspect="auto",
            origin="lower",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        )

        # 設定座標軸範圍與光譜圖一致
        self.ax.set_xlim(flxs[0], flxs[-1])
        self.ax.set_ylim(fpts[0], fpts[-1])

        # 散點圖
        self.scatter = self.ax.scatter(s_flxs, s_fpts, color=self.colors, s=2)

        # 保存原始數據
        self.s_flxs = s_flxs
        self.s_fpts = s_fpts

        # 創建選擇圓圈（初始設為不可見）
        # 因為x, y 長度不同，所以改用Ellipse
        self.select_x = 0.03 * (flxs[-1] - flxs[0])
        self.select_y = 0.03 * (fpts[-1] - fpts[0])
        self.circle = patches.Ellipse(
            (0, 0),
            self.select_x,
            self.select_y,
            fill=False,
            color=self.current_color,
            linestyle="--",
            visible=False,
        )
        self.ax.add_patch(self.circle)

        # 添加按鈕
        self.add_buttons()

        # 連接事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # 創建動畫
        self.anim = FuncAnimation(
            self.fig,
            self.update_animation,
            interval=33,  # 約30 FPS
            blit=True,
            cache_frame_data=False,
        )

    def add_buttons(self):
        from matplotlib.widgets import Button

        # 創建按鈕
        ax_red = plt.axes([0.7, 0.05, 0.1, 0.04])
        ax_black = plt.axes([0.81, 0.05, 0.1, 0.04])
        ax_done = plt.axes([0.92, 0.05, 0.1, 0.04])

        self.btn_red = Button(ax_red, "Red", color="lightcoral")
        self.btn_black = Button(ax_black, "Black", color="gray")
        self.btn_done = Button(ax_done, "Done", color="lightgreen")

        self.btn_red.on_clicked(lambda event: self.set_color("r"))
        self.btn_black.on_clicked(lambda event: self.set_color("k"))
        self.btn_done.on_clicked(self.finish_selection)

    def set_color(self, color):
        self.current_color = color
        self.is_selecting = True
        self.circle.set_color(color)

    def update_animation(self, frame):
        if not self.is_selecting or self.mouse_x is None or self.mouse_y is None:
            self.circle.set_visible(False)
            return [self.circle]

        self.circle.center = (self.mouse_x, self.mouse_y)
        self.circle.set_visible(True)
        return [self.circle]

    def on_motion(self, event):
        if not event.inaxes == self.ax:
            self.mouse_x = None
            self.mouse_y = None
            return

        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

        # 如果滑鼠正在拖曳中，就更新點的顏色
        if self.is_dragging:
            self.update_points(event.xdata, event.ydata)

    def update_points(self, x, y):
        # 計算點擊位置附近的點
        distances = ((self.s_flxs - x) / self.select_x) ** 2 + (
            (self.s_fpts - y) / self.select_y
        ) ** 2
        mask = distances < 0.25

        # 更新顏色
        self.colors[mask] = self.current_color
        self.scatter.set_color(self.colors)

    def on_press(self, event):
        if not self.is_selecting or event.inaxes != self.ax:
            return

        self.is_dragging = True
        self.update_points(event.xdata, event.ydata)

    def on_release(self, event):
        self.is_dragging = False

    def show_and_run(self):
        plt.show()

    def finish_selection(self, event):
        # 停止動畫
        self.anim.event_source.stop()

    def get_selected_points(self):
        self.finish_selection(None)
        plt.close(self.fig)
        mask = self.colors == "r"
        s_flxs = self.s_flxs[mask]
        s_fpts = self.s_fpts[mask]
        return s_flxs, s_fpts, self.colors


class InteractiveLines:
    def __init__(self, spectrum, flxs, fpts, cflx=None, eflx=None):
        from matplotlib.animation import FuncAnimation

        self.fig, self.ax = plt.subplots()
        self.flxs = flxs

        # 顯示光譜圖
        self.ax.imshow(
            spectrum,
            aspect="auto",
            origin="lower",
            extent=(flxs[0], flxs[-1], fpts[0], fpts[-1]),
        )

        # 初始化線的位置
        self.cflx = (flxs[0] + flxs[-1]) / 2 if cflx is None else cflx
        self.eflx = flxs[-5] if eflx is None else eflx

        # 創建兩條垂直線
        self.line1 = self.ax.axvline(x=self.cflx, color="r", linestyle="--", picker=5)
        self.line2 = self.ax.axvline(x=self.eflx, color="b", linestyle="--", picker=5)

        # 設置變數
        self.picked = None
        self.min_dist = 0.1 * (flxs[-1] - flxs[0])
        self.mouse_x = None
        self.is_finished = False

        # 添加完成按鈕
        self.add_finish_button()

        # 連接事件
        self.fig.canvas.mpl_connect("pick_event", self.onpick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)

        # xlim, ylim
        self.ax.set_xlim(flxs[0], flxs[-1])
        self.ax.set_ylim(fpts[0], fpts[-1])

        # 創建動畫
        self.anim = FuncAnimation(
            self.fig,
            self.update_animation,
            interval=33,  # 約30 FPS
            blit=True,
            cache_frame_data=False,
        )

        plt.show()

    def add_finish_button(self):
        from matplotlib.widgets import Button

        # 創建按鈕的軸域，位置在圖表右下角
        ax_finish = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.btn_finish = Button(ax_finish, "Finish", color="lightgreen")
        self.btn_finish.on_clicked(self.on_finish)

    def onpick(self, event):
        if self.is_finished:
            return
        # 檢查滑鼠點擊是否在線上
        if event.mouseevent.name != "button_press_event":
            return

        # 切換選中狀態
        if self.picked is None:
            self.picked = event.artist
        else:
            self.picked = None

    def onmove(self, event):
        if self.is_finished:
            return
        if event.inaxes != self.ax:
            self.mouse_x = None
            return
        self.mouse_x = event.xdata

    def update_animation(self, frame):
        if self.picked is None or self.mouse_x is None:
            return [self.line1, self.line2]

        new_x = self.mouse_x
        other_line = self.line2 if self.picked is self.line1 else self.line1
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

        return [self.line1, self.line2]

    def on_finish(self, event):
        """完成按鈕的回調函數"""
        self.is_finished = True
        self.picked = None
        # 停止動畫
        self.anim.event_source.stop()

    def get_positions(self):
        """運行交互式選擇器並返回兩條線的位置"""
        self.on_finish(None)
        plt.close(self.fig)
        return float(self.line1.get_xdata()[0]), float(self.line2.get_xdata()[0])


def calculate_energy(flxs, EJ, EC, EL, cutoff=50):
    from scqubits import Fluxonium

    fluxonium = Fluxonium(EJ, EC, EL, flux=0.0, cutoff=cutoff, truncated_dim=10)
    spectrumData = fluxonium.get_spectrum_vs_paramvals("flux", flxs, evals_count=10)

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


def spectrum_analyze(flxs, fpts, signals, ratio, min_dist=None):
    amps = np.abs(signals - np.mean(signals, axis=0, keepdims=True))
    amps /= np.std(amps, axis=0, keepdims=True)

    # find peaks
    fpt_idxs = np.argmax(amps, axis=0)  # (len(flxs),)
    maxs = amps[fpt_idxs, np.arange(amps.shape[1])]  # (len(flxs),)
    masks = maxs >= ratio  # (len(flxs),)
    s_flxs = flxs[masks]
    s_fpts = fpts[fpt_idxs][masks]

    # mask +/- 5 points around the selected points
    # and try to find the second peak
    d_idxs = np.maximum(0, fpt_idxs - 5)
    u_idxs = np.minimum(len(fpts), fpt_idxs + 6)
    for i in range(len(fpt_idxs)):
        amps[d_idxs[i] : u_idxs[i], i] = 0
    fpt_idxs = np.argmax(amps, axis=0)
    maxs = amps[fpt_idxs, np.arange(amps.shape[1])]
    masks = maxs >= ratio
    s_flxs2 = flxs[masks]
    s_fpts2 = fpts[fpt_idxs][masks]

    # append the second peaks
    s_flxs = np.concatenate([s_flxs, s_flxs2])
    s_fpts = np.concatenate([s_fpts, s_fpts2])

    if min_dist is not None:
        s_flxs, s_fpts = remove_close_points(s_flxs, s_fpts, min_dist)

    return s_flxs, s_fpts


def remove_close_points(flxs, fpts, dist_ratio):
    # remove some close points
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
    for i, j in allows["transitions"]:
        fs.append(energies[:, j] - energies[:, i])
        labels.append((i, j))
        names.append(f"{i} -> {j}")
    for i, j in allows["mirror"]:
        freq = energies[:, j] - energies[:, i]
        fs.append(2 * allows["sample_f"] - freq)
        labels.append((j, i))
        names.append(f"{i} -> {j} mirror")

    return np.array(fs).T, np.array(labels), names


def search_in_database(flxs, fpts, datapath, allows):
    from h5py import File

    with File(datapath, "r") as file:
        h_flxs = file["flxs"][:]  # type: ignore
        h_params = file["params"][:]  # type: ignore
        h_energies = file["energies"][:]  # type: ignore

    def dist_to_curve(energies):
        fs, *_ = energy2transition(energies, allows)

        dist = np.abs(fs - fpts[:, None])  # (n, m)
        dist = np.nanmin(dist, axis=1)  # (n, )

        return np.nansum(dist)

    # find the closest index in energy to s_fs
    flxs = np.mod(flxs, 1.0)
    d2 = (h_flxs[:, None] - flxs[None, :]) ** 2  # (m, m')
    idxs = np.argmin(d2, axis=0)  # (m', )

    best_params = None
    best_energy = None
    best_dist = float("inf")
    for i in trange(h_params.shape[0]):
        energy = h_energies[i, idxs]

        dist = dist_to_curve(energy)
        if dist < best_dist:
            best_dist = dist
            best_energy = h_energies[i]
            best_params = h_params[i]

    return best_params, h_flxs, best_energy


def fit_spectrum(flxs, fpts, init_params, allows, params_b=None, maxfun=1000):
    import scqubits as scq
    from scipy.optimize import minimize

    scq.settings.PROGRESSBAR_DISABLED, old = True, scq.settings.PROGRESSBAR_DISABLED

    max_level = int(
        np.max(
            [
                np.max(np.array(allows["transitions"]), initial=0),
                np.max(np.array(allows["mirror"]), initial=0),
            ]
        )
        + 1
    )
    fluxonium = scq.Fluxonium(
        *init_params, flux=0.0, truncated_dim=max_level, cutoff=40
    )

    pbar = tqdm(
        desc=f"({init_params[0]:.2f}, {init_params[1]:.2f}, {init_params[2]:.2f})",
        total=maxfun,
    )

    def callback(intermediate_result):
        pbar.update(1)
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
        min_dist2 = dist2[idxs, min_idx]
        grad_energies = np.zeros_like(energies)
        for i, (f, t) in enumerate(labels[min_idx]):
            grad_energies[i, f] -= 2 * dist[i, min_idx[i]]
            grad_energies[i, t] += 2 * dist[i, min_idx[i]]

        return np.sum(min_dist2), grad_energies

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

    return res.x
