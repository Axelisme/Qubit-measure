import numpy as np
from scipy.linalg import expm
from typing import Tuple
from numpy.typing import NDArray

from .base import fit_func


def fit_transition_rates(
    times: NDArray[np.float64], populations: NDArray[np.float64]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    擬合三能階系統 (G, E, O) 的躍遷速率。

    Args:
        times: 時間點陣列 (1D array)
        populations_over_time: 對應時間點的分佈陣列 (N x 3), 順序為 [pg, pe, po]

    Returns:
        fitted_initial_distribution: (pg_0, pe_0, po_0)
        fitted_rates: (T_ge, T_eg, T_eo, T_og)
    """

    # 展平數據以供 curve_fit 使用 (N, 3) -> (3N,)
    y_data_flat = populations.flatten()

    # 2. 定義模型函數
    # -----------------------------------------------------
    def model_func(t, T_ge, T_eg, T_eo, T_og, pg0, pe0):
        p_ge = pg0 + pe0
        if p_ge > 1.0:
            pg0 /= p_ge
            pe0 /= p_ge

        po0 = 1.0 - pg0 - pe0

        P0 = np.array([pg0, pe0, po0])

        # 建立速率矩陣 M
        # M 定义: dP/dt = M * P
        M = np.array(
            [
                [-T_ge, T_eg, T_og],
                [T_ge, -(T_eg + T_eo), 0],
                [0, T_eo, -T_og],
            ]
        )

        # 計算演化
        P_t = np.array([expm(M * time_point) @ P0 for time_point in t])

        return P_t.flatten()

    # 3. 初始化 (Initialization)
    init_pg, init_pe, _ = populations[0]

    # B. 估計初始斜率 (使用前 5 點做簡單線性回歸以抗噪)
    n_slope = min(5, len(times))
    dt_slope = times[1] - times[0]
    # 簡單差分: slope ~ (P[n] - P[0]) / (t[n] - t[0])
    # 這裡只取前幾個點的平均斜率
    slope_g = np.mean(np.diff(populations[:n_slope, 0]) / dt_slope)
    slope_o = np.mean(np.diff(populations[:n_slope, 2]) / dt_slope)

    # 假設主要從 E 出發 (pe ~ 1)，則斜率直接反映速率
    # 加上 max(..., 1e-6) 避免 0 或負值導致 log 錯誤或鎖死
    guess_T_eg = max(slope_g, 1e-4)  # G 增加的速率
    guess_T_eo = max(slope_o, 1e-4)  # O 增加的速率

    # C. 估計穩態與其餘速率
    # 取最後 10% 的數據作為穩態平均
    n_steady = max(1, int(len(times) * 0.1))
    P_steady = np.mean(populations[-n_steady:], axis=0)
    pg_inf, pe_inf, po_inf = P_steady

    # 避免除以零
    pg_inf = max(pg_inf, 1e-9)
    pe_inf = max(pe_inf, 1e-9)
    po_inf = max(po_inf, 1e-9)

    # 利用流量平衡推導 T_ge, T_og
    # T_og * po_inf = T_eo * pe_inf  => T_og = T_eo * (pe/po)
    guess_T_og = guess_T_eo * (pe_inf / po_inf)

    # T_ge * pg_inf = (T_eg + T_eo) * pe_inf => T_ge = (T_eg + T_eo) * (pe/pg)
    guess_T_ge = (guess_T_eg + guess_T_eo) * (pe_inf / pg_inf)

    p0_guess = [guess_T_ge, guess_T_eg, guess_T_eo, guess_T_og, init_pg, init_pe]

    # 4. 執行 Curve Fit
    # -----------------------------------------------------
    # Bounds: Rates >= 0, 0 <= Prob <= 1
    # 參數順序: T_ge, T_eg, T_eo, T_og, pg0, pe0
    lower_bounds = [0, 0, 0, 0, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, 1, 1]

    pOpt, _ = fit_func(
        times, y_data_flat, model_func, p0_guess, bounds=(lower_bounds, upper_bounds)
    )

    # 5. 整理輸出
    # -----------------------------------------------------
    res_T_ge, res_T_eg, res_T_eo, res_T_og, res_pg0, res_pe0 = pOpt
    res_po0 = 1.0 - res_pg0 - res_pe0

    fitted_initial = (res_pg0, res_pe0, res_po0)
    fitted_rates = (res_T_ge, res_T_eg, res_T_eo, res_T_og)

    return fitted_initial, fitted_rates
