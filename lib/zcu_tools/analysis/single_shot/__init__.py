from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Circle

from ..fitting import fit_gauss_2d, fit_gauss_2d_bayesian
from .ge import fidelity_func, singleshot_ge_analysis, singleshot_visualize
from .rabi import singleshot_rabi_analysis, visualize_singleshot_rabi


def fit_singleshot2d(
    signals: np.ndarray,
    num_gauss: int = 3,
    method: Literal["standard", "bayesian"] = "baayesian",
) -> np.ndarray:
    """
    使用二維高斯混合模型擬合單次測量數據並視覺化結果。

    參數
    ----------
    signals : np.ndarray
        複數測量信號。可以是一維數組或二維數組（多個測量組）。
    num_gauss : int, default=2
        要擬合的高斯分佈數量。
    plot : bool, default=True
        是否生成可視化圖形。
    method : Literal["standard", "bayesian"], default="standard"
        使用的擬合方法:
        - "standard": 使用標準GMM擬合指定數量的高斯分佈
        - "bayesian": 使用貝葉斯GMM自動確定最佳組件數量（不超過num_gauss）
    現在總是繪製信心圓，半徑由 sigma 決定。

    返回
    -------
    np.ndarray
        形狀為(m, 4)的擬合結果，每行包含一個高斯分佈的參數 [x0, y0, sigma, n]，
        其中m是擬合的高斯分佈數量，可能小於等於num_gauss。
    """
    # 獲取信號的實部和虛部 (I和Q)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    # 將多組信號合併為單一數據集
    all_signals = np.concatenate(signals) if signals.shape[0] > 1 else signals[0]
    xs = all_signals.real
    ys = all_signals.imag

    MAX_FIT_POINTS = 1e5
    if len(xs) > MAX_FIT_POINTS:
        idx = np.random.choice(len(xs), int(MAX_FIT_POINTS), replace=False)
        fit_xs = xs[idx]
        fit_ys = ys[idx]
    else:
        fit_xs = xs
        fit_ys = ys

    # 根據方法選擇擬合函數
    if method == "standard":
        params = fit_gauss_2d(fit_xs, fit_ys, num_gauss=num_gauss)
    elif method == "bayesian":
        params = fit_gauss_2d_bayesian(fit_xs, fit_ys, num_gauss=num_gauss)
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. Use 'standard' or 'bayesian'."
        )

    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 為每個數據點分配最接近的高斯分佈
    distances = np.zeros((len(xs), len(params)))
    for i, (x0, y0, sigma, _) in enumerate(params):
        # 計算每個點到各高斯中心的歸一化距離
        distances[:, i] = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / sigma

    # 獲得每個點最近的高斯分佈索引
    closest_gaussian = np.argmin(distances, axis=1)

    # 繪製數據點，按照最近的高斯分佈著色
    colors = list(TABLEAU_COLORS.values())

    # 限制最大顯示點數以提高性能
    MAX_POINTS = 1e5
    if len(xs) > MAX_POINTS:
        idx = np.random.choice(len(xs), int(MAX_POINTS), replace=False)
        plot_xs = xs[idx]
        plot_ys = ys[idx]
        plot_closest = closest_gaussian[idx]
    else:
        plot_xs = xs
        plot_ys = ys
        plot_closest = closest_gaussian

    # 繪製數據點
    for i in range(len(params)):
        mask = plot_closest == i
        if np.any(mask):
            ax.scatter(
                plot_xs[mask],
                plot_ys[mask],
                marker=".",
                alpha=0.3,
                edgecolor="None",
                c=colors[i % len(colors)],
                label=f"Cluster {i + 1} ({params[i, 3]:.1%})",
            )

    # 標記高斯分佈中心
    for i, (x0, y0, sigma, weight) in enumerate(params):
        ax.plot(
            x0,
            y0,
            linestyle="",
            marker="o",
            markersize=10,
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor="k",
            markeredgewidth=1.5,
        )

        # 繪製信心圓 (各向同性，sigma 為半徑)
        circle = Circle(
            xy=(x0, y0),
            radius=sigma,
            edgecolor=colors[i % len(colors)],
            fc="None",
            lw=2,
            alpha=0.7,
            linestyle="--",
        )
        ax.add_patch(circle)

    ax.set_title(f"Singleshot Fitting ({method})")
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    ax.legend(loc="best")
    ax.axis("equal")

    return params


__all__ = [
    "fidelity_func",
    "singleshot_ge_analysis",
    "fit_singleshot2d",
    "singleshot_visualize",
    "singleshot_rabi_analysis",
    "visualize_singleshot_rabi",
]
