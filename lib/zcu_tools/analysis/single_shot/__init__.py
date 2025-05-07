from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

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

    MAX_FIT_POINTS = 1e4
    if len(xs) > MAX_FIT_POINTS:
        idx = np.random.choice(len(xs), int(MAX_FIT_POINTS), replace=False)
        fit_xs = xs[idx]
        fit_ys = ys[idx]
    else:
        fit_xs = xs
        fit_ys = ys

    # 根據方法選擇擬合函數
    if method == "standard":
        params, gmm = fit_gauss_2d(fit_xs, fit_ys, num_gauss=num_gauss)
    elif method == "bayesian":
        params, gmm = fit_gauss_2d_bayesian(fit_xs, fit_ys, num_gauss=num_gauss)
    else:
        raise ValueError(
            f"Unknown fitting method: {method}. Use 'standard' or 'bayesian'."
        )

    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 繪製數據點，所有點使用統一黑色
    colors = list(TABLEAU_COLORS.values())

    # 限制最大顯示點數以提高性能
    MAX_POINTS = 1e5
    if len(xs) > MAX_POINTS:
        idx = np.random.choice(len(xs), int(MAX_POINTS), replace=False)
        plot_xs = xs[idx]
        plot_ys = ys[idx]
    else:
        plot_xs = xs
        plot_ys = ys

    # 根據GMM分配每個點的高斯分佈標籤
    labels = gmm.predict(np.column_stack([plot_xs, plot_ys]))

    # 按高斯分佈分組繪製點
    for i in range(len(params)):
        mask = labels == i
        ax.scatter(
            plot_xs[mask],
            plot_ys[mask],
            marker=".",
            alpha=0.1,
            edgecolor="None",
            c=colors[(i + 1) % len(colors)],
            label=f"Data points (G{i + 1})" if i == 0 else None,
        )

    # 標記高斯分佈中心
    for i, (x0, y0, _, weight) in enumerate(params):
        ax.plot(
            x0,
            y0,
            linestyle="",
            marker="o",
            markersize=10,
            markerfacecolor=colors[i % len(colors) + 1],
            markeredgecolor="k",
            markeredgewidth=1.5,
            label=f"Gaussian {i + 1}: {weight:.2%}",
        )

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
