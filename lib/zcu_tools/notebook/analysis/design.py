from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scqubits as scq
from tqdm.auto import tqdm

from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate.fluxonium import (
    calculate_dispersive,
    calculate_dispersive_sweep,
)


def generate_params_table(
    EJ: Union[float, np.ndarray, Tuple[float, float]],
    EC: Union[float, np.ndarray, Tuple[float, float]],
    EL: Union[float, np.ndarray, Tuple[float, float]],
    flx: float = 0.5,
) -> pd.DataFrame:
    """
    Create a table with columns: flx, EJ, EC, EL

    Args:
        EJ: EJ float or np.ndarray
        EC: EC float or np.ndarray
        EL: EL float or np.ndarray
        flx: flux value

    Returns:
        DataFrame with columns: flx, EJ, EC, EL
    """

    if isinstance(EJ, float):
        EJ = np.array([EJ])
    elif isinstance(EJ, tuple):
        EJ = np.arange(EJ[0], EJ[1], 0.05)

    if isinstance(EC, float):
        EC = np.array([EC])
    elif isinstance(EC, tuple):
        EC = np.arange(EC[0], EC[1], 0.05)

    if isinstance(EL, float):
        EL = np.array([EL])
    elif isinstance(EL, tuple):
        EL = np.arange(EL[0], EL[1], 0.05)

    return pd.DataFrame(
        [
            {
                "flx": flx,
                "EJ": eJ,
                "EC": eC,
                "EL": eL,
            }
            for eJ, eC, eL in product(EJ, EC, EL)
        ]
    )


def calculate_esys(params_table: pd.DataFrame, fluxonium: scq.Fluxonium) -> None:
    """
    計算每個參數組合下的 fluxonium 能譜

    會在 params_table 中新增一個 "esys" 欄位
    """

    def calc_single_esys(row):
        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]
        return fluxonium.eigensys(evals_count=fluxonium.truncated_dim)

    tqdm.pandas()
    params_table["esys"] = params_table.progress_apply(calc_single_esys, axis=1)


def calculate_f01(params_table: pd.DataFrame, fluxonium: scq.Fluxonium) -> None:
    """
    計算每個參數組合下的 f01, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "f01" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    def calc_single_f01(row):
        evals, _ = row["esys"]
        return evals[1] - evals[0]

    params_table["f01"] = params_table.apply(calc_single_f01, axis=1)


def calculate_m01(params_table: pd.DataFrame, fluxonium: scq.Fluxonium) -> None:
    """
    計算每個參數組合下的 m01, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "m01" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    def calc_single_m01(row):
        return np.abs(fluxonium.n_operator(energy_esys=row["esys"])[0, 1])

    params_table["m01"] = params_table.apply(calc_single_m01, axis=1)


def calculate_dipersive_shift(params_table: pd.DataFrame, g: float, r_f: float) -> None:
    params_list = params_table.to_dict(orient="records")

    def update_fn(fluxonium: scq.Fluxonium, row: Dict[str, Any]) -> None:
        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]

    rf_0, rf_1 = calculate_dispersive_sweep(
        params_list, update_fn, g, r_f, evals_count=20, progress=True
    )

    params_table["Chi"] = np.abs(rf_0 - rf_1)


def calculate_t1(
    params_table: pd.DataFrame,
    fluxonium: scq.Fluxonium,
    noise_channels: List[Tuple[str, Dict[str, Any]]],
    Temp: float,
) -> None:
    """
    計算每個參數組合下的 t1, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "t1" 欄位
    """

    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    # Suppress the warning when calculating t1
    old, scq.settings.T1_DEFAULT_WARNING = scq.settings.T1_DEFAULT_WARNING, False

    def calc_single_t1(row):
        return fluxonium.t1_effective(
            noise_channels=noise_channels,
            common_noise_options=dict(i=1, j=0, T=Temp),
            esys=row["esys"],
        )

    params_table["t1"] = params_table.apply(calc_single_t1, axis=1)

    scq.settings.T1_DEFAULT_WARNING = old


def calculate_collision(params_table: pd.DataFrame, avoid_freqs: List[float]) -> None:
    """
    計算每個參數組合下的 collision, 需要已經計算過 esys 的params_table

    會在 params_table 中新增一個 "collision" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    COLLI_THRESHOLD = 0.3

    freqs = np.array(avoid_freqs)[None, :]

    def calc_single_collision(row):
        evals, _ = row["esys"]
        e0x = evals - evals[0]
        e1x = evals - evals[1]

        # 檢查能階差與避免頻率的差距是否小於閾值
        e0x_collision = np.min(np.abs(e0x[:, None] - freqs), axis=0) < COLLI_THRESHOLD
        e1x_collision = np.min(np.abs(e1x[:, None] - freqs), axis=0) < COLLI_THRESHOLD

        return np.any(e0x_collision | e1x_collision)

    params_table["collision"] = params_table.apply(calc_single_collision, axis=1)


def plot_scan_results(params_table: pd.DataFrame) -> go.Figure:
    plot_table = params_table.drop("esys", axis=1)
    plot_table["Label"] = plot_table.apply(
        lambda row: f"EJ={row['EJ']:.2f}, EC={row['EC']:.3f}, EL={row['EL']:.3f}",
        axis=1,
    )

    # 繪製散點圖
    fig = px.scatter(
        plot_table,
        x="Chi",
        y="t1",
        color="collision",
        color_discrete_map={True: "red", False: "blue"},
        log_x=True,
        log_y=True,
        hover_name="Label",
        labels={"Chi": "Chi", "T1 (us)": "t1"},
    )
    fig.update_traces(marker=dict(size=3))

    # 預設隱藏collision=True的點
    fig.for_each_trace(
        lambda trace: trace.update(visible="legendonly") if trace.name == "True" else ()
    )

    fig.update_layout(
        xaxis_title="Chi",
        yaxis_title="T1 (ns)",
        title_x=0.501,
        template="plotly_white",
        showlegend=True,
        width=1100,
        height=750,
    )
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")

    return fig


def annotate_best_point(fig, data: pd.DataFrame) -> Tuple[float, float, float]:
    return None, None, None


def add_real_sample(
    fig: go.Figure, result_path: str, t1: float, t1_info, chi_info, flx: float = 0.5
) -> None:
    name, param, *_ = load_result(result_path)

    r_f, g = chi_info
    rf_0, rf_1 = calculate_dispersive(param, r_f, g, flx, cutoff=40, evals_count=20)
    chi = np.abs(rf_0 - rf_1)

    scq.settings.T1_DEFAULT_WARNING = False

    noise_channels, Temp = t1_info
    fluxonium = scq.Fluxonium(*param, flux=flx, cutoff=40, truncated_dim=2)
    predict_t1 = 1e-3 * fluxonium.t1_effective(
        noise_channels=noise_channels,
        common_noise_options=dict(i=1, j=0, T=Temp),
    )

    # 添加從實際t1到預測t1的線段
    fig.add_shape(
        type="line",
        x0=chi,
        y0=1e3 * t1,
        x1=chi,
        y1=1e3 * predict_t1,
        line=dict(color="red", width=1, dash="dot"),
    )

    # 添加實際t1的點
    fig.add_scatter(
        x=[chi],
        y=[1e3 * t1],
        mode="markers+text",
        marker=dict(symbol="x", size=5, color="black"),
        text=[name],
        textposition="top center",
        showlegend=False,
    )

    # 添加預測t1的點
    fig.add_scatter(
        x=[chi],
        y=[1e3 * predict_t1],
        mode="markers",
        marker=dict(symbol="x", size=5, color="red"),
        showlegend=False,
    )
