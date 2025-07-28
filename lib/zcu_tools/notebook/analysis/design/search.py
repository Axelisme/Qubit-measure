import os
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scqubits as scq
from tqdm.auto import tqdm

from zcu_tools.notebook.persistance import load_result
from zcu_tools.simulate.fluxonium import calculate_chi_sweep, calculate_dispersive

DESIGN_CUTOFF = 40
DESIGN_EVALS_COUNT = 15


def generate_params_table(
    EJ: Union[float, np.ndarray, Tuple[float, float]],
    EC: Union[float, np.ndarray, Tuple[float, float]],
    EL: Union[float, np.ndarray, Tuple[float, float]],
    flx: float = 0.5,
    precision: float = 0.1,
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
        EJ = np.arange(EJ[0], EJ[1], precision)

    if isinstance(EC, float):
        EC = np.array([EC])
    elif isinstance(EC, tuple):
        EC = np.arange(EC[0], EC[1], precision)

    if isinstance(EL, float):
        EL = np.array([EL])
    elif isinstance(EL, tuple):
        EL = np.arange(EL[0], EL[1], precision)

    return pd.DataFrame(
        [
            {
                "flx": flx,
                "EJ": eJ,
                "EC": eC,
                "EL": eL,
                "valid": True,
            }
            for eJ, eC, eL in product(EJ, EC, EL)
        ]
    )


def calculate_esys(params_table: pd.DataFrame) -> None:
    """
    計算每個參數組合下的 fluxonium 能譜

    會在 params_table 中新增一個 "esys" 欄位
    """

    fluxonium = scq.Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    def calc_single_esys(row):
        nonlocal fluxonium

        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]
        return fluxonium.eigensys(evals_count=fluxonium.truncated_dim)

    tqdm.pandas()
    params_table["esys"] = params_table.progress_apply(calc_single_esys, axis=1)


def calculate_f01(params_table: pd.DataFrame) -> None:
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


def calculate_m01(params_table: pd.DataFrame) -> None:
    """
    計算每個參數組合下的 m01, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "m01" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    fluxonium = scq.Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    def calc_single_m01(row):
        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]

        return np.abs(fluxonium.n_operator(energy_esys=row["esys"])[0, 1])

    params_table["m01"] = params_table.apply(calc_single_m01, axis=1)


def calculate_dipersive_shift(params_table: pd.DataFrame, g: float, r_f: float) -> None:
    params_list = params_table.to_dict(orient="records")

    def update_fn(fluxonium: scq.Fluxonium, row: Dict[str, Any]) -> None:
        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]

    chis = calculate_chi_sweep(params_list, update_fn, g, r_f, progress=True)
    params_table["Chi"] = np.abs(chis[:, 1] - chis[:, 0])


def calculate_t1(
    params_table: pd.DataFrame,
    noise_channels: List[Tuple[str, Dict[str, Any]]],
    Temp: float,
) -> None:
    """
    計算每個參數組合下的 t1, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "t1" 欄位
    """

    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    fluxonium = scq.Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    # Suppress the warning when calculating t1
    old, scq.settings.T1_DEFAULT_WARNING = scq.settings.T1_DEFAULT_WARNING, False

    def calc_single_t1(row):
        fluxonium.flux = row["flx"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]

        return fluxonium.t1_effective(
            noise_channels=noise_channels,
            common_noise_options=dict(i=1, j=0, T=Temp),
            esys=row["esys"],
        )

    params_table["t1"] = params_table.apply(calc_single_t1, axis=1)

    scq.settings.T1_DEFAULT_WARNING = old


def avoid_collision(
    params_table: pd.DataFrame, avoid_freqs: List[float], threshold: float = 0.3
) -> None:
    """
    計算每個參數組合下的 collision, 需要已經計算過 esys 的params_table

    會在 params_table 中新增一個 "collision" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    freqs = np.array(avoid_freqs)[None, :]

    def calc_single_collision(row):
        evals, _ = row["esys"]
        e0x = evals - evals[0]
        e1x = evals - evals[1]

        # 檢查能階差與避免頻率的差距是否小於閾值
        e0x_collision = np.min(np.abs(e0x[:, None] - freqs), axis=0) < threshold
        e1x_collision = np.min(np.abs(e1x[:, None] - freqs), axis=0) < threshold

        return np.any(e0x_collision | e1x_collision)

    params_table["collision"] = params_table.apply(calc_single_collision, axis=1)
    params_table["valid"] &= ~params_table["collision"]


def avoid_low_f01(params_table: pd.DataFrame, f01_threshold: float) -> None:
    """
    移除 f01 小於 f01_threshold 的參數組合
    """

    params_table["low_f01"] = params_table["f01"] < f01_threshold
    params_table["valid"] &= ~params_table["low_f01"]


def avoid_low_m01(params_table: pd.DataFrame, m01_threshold: float) -> None:
    """
    移除 m01 小於 m01_threshold 的參數組合
    """

    params_table["low_m01"] = params_table["m01"] < m01_threshold
    params_table["valid"] &= ~params_table["low_m01"]


def plot_scan_results(params_table: pd.DataFrame) -> go.Figure:
    # Remove the heavy esys column if present; ignore errors to stay robust when it is
    # already absent.
    plot_table = params_table.drop(columns=["esys"], errors="ignore")

    # Helper to build a descriptive label that conditionally includes additional flags
    def _build_label(row: pd.Series) -> str:
        """Return a label string for hover info.

        It always shows EJ, EC and EL, and conditionally appends the state of
        optional columns (collision, low_f01, low_m01) only when they exist in
        the dataframe.
        """

        parts = [
            f"EJ={row['EJ']:.2f}",
            f"EC={row['EC']:.3f}",
            f"EL={row['EL']:.3f}",
        ]

        for opt_flag in ("collision", "low_f01", "low_m01"):
            if opt_flag in row.index:
                parts.append(f"{opt_flag}={row[opt_flag]}")

        return ", ".join(parts)

    plot_table["Label"] = plot_table.apply(_build_label, axis=1)

    # 繪製散點圖
    fig = px.scatter(
        plot_table,
        x="Chi",
        y="t1",
        color="valid",
        color_discrete_map={True: "blue", False: "red"},
        log_x=True,
        log_y=True,
        hover_name="Label",
        labels={"Chi": "Chi", "T1 (us)": "t1"},
    )
    fig.update_traces(marker=dict(size=3))

    # 預設隱藏valid=False的點
    fig.for_each_trace(
        lambda trace: trace.update(visible="legendonly")
        if trace.name == "False"
        else ()
    )

    fig.update_layout(
        title_x=0.501,
        xaxis_title="Chi (GHz)",
        yaxis_title="T1 (ns)",
        template="plotly_white",
        showlegend=True,
        width=1100,
        height=750,
    )
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")

    return fig


def annotate_best_point(fig, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Find and plot the best snr's param on the plot, the equation of snr is:
        snr = Chi * sqrt(T1)
    """

    # filter out invalid params
    valid_data = data[data["valid"]]

    snrs = valid_data["Chi"] * np.sqrt(valid_data["t1"])
    best_param = valid_data.iloc[np.argmax(snrs)]

    EJ, EC, EL = (
        float(best_param["EJ"]),
        float(best_param["EC"]),
        float(best_param["EL"]),
    )

    fig.add_annotation(
        x=np.log10(best_param["Chi"]),
        y=np.log10(best_param["t1"]),
        text=f"{EJ:.2f}/{EC:.2f}/{EL:.2f}",
        showarrow=True,
        arrowhead=1,
        arrowcolor="black",
        ax=0,
        ay=-20,
    )

    # draw a line of same snr across best_param
    # xs = np.linspace(valid_data["Chi"].min(), valid_data["Chi"].max(), 100)
    # ys = (best_param["Chi"] * np.sqrt(best_param["t1"]) / xs) ** 2
    # fig.add_trace(
    #     go.Scatter(
    #         x=xs,
    #         y=ys,
    #         mode="lines",
    #         name="Same SNR",
    #         line=dict(color="black", width=1, dash="dot"),
    #     )
    # )

    return EJ, EC, EL


def add_real_sample(
    fig: go.Figure,
    chip_name: str,
    noise_channels: List[Tuple[str, Dict[str, Any]]],
    Temp: float,
    flx: float = 0.5,
    result_dir: Optional[str] = None,
) -> None:
    """
    Add a real chip sample to the plot

    Args:
        fig: plotly figure
        chip_name: chip name
        t1_info: t1 info
        flx: flux
        result_dir: result directory, default to "../../result"
    """
    if result_dir is None:
        result_dir = os.path.join("..", "..", "result")

    result_path = os.path.join(result_dir, chip_name, "params.json")
    _, param, mA_c, _, _, result = load_result(result_path)

    # unpack result
    r_f, g = result["dispersive"]["r_f"], result["dispersive"]["g"]

    # load freq data
    freq_path = os.path.join(result_dir, chip_name, "sample.csv")
    freq_df = pd.read_csv(freq_path)
    idx = np.argmin(np.abs(freq_df["calibrated mA"] - mA_c))
    t1 = freq_df["T1 (us)"].iloc[idx]

    # calculate chi
    rf_0, rf_1 = calculate_dispersive(param, flx, r_f, g)
    chi = np.abs(rf_0 - rf_1)

    # calculate t1
    fluxonium = scq.Fluxonium(*param, flux=flx, cutoff=DESIGN_CUTOFF, truncated_dim=2)
    scq.settings.T1_DEFAULT_WARNING = False
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
        name=chip_name,
        legendgroup=chip_name,
        showlegend=True,
    )

    # 添加實際t1的點
    fig.add_scatter(
        x=[chi],
        y=[1e3 * t1],
        mode="markers+text",
        marker=dict(symbol="x", size=5, color="black"),
        text=[chip_name],
        textposition="top center",
        hovertemplate=f"<b>{chip_name}</b><br>"
        + f"χ: {chi:.2f} MHz<br>"
        + f"T1: {t1:.2f} us<br>"
        + f"EJ: {param[0]:.3f} GHz<br>"
        + f"EC: {param[1]:.3f} GHz<br>"
        + f"EL: {param[2]:.3f} GHz",
        legendgroup=chip_name,
        showlegend=False,
    )

    # 添加預測t1的點
    fig.add_scatter(
        x=[chi],
        y=[1e3 * predict_t1],
        mode="markers",
        marker=dict(symbol="x", size=5, color="red"),
        legendgroup=chip_name,
        showlegend=False,
    )
