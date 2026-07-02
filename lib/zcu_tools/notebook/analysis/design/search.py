from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.meta_tool import QubitParams
from zcu_tools.simulate.fluxonium import calculate_chi_sweep
from zcu_tools.simulate.fluxonium.branch.floquet import calc_ge_snr

DESIGN_CUTOFF = 40
DESIGN_EVALS_COUNT = 15


ParamGridInput = float | np.ndarray | tuple[float, float]


@contextmanager
def _t1_default_warning_disabled() -> Iterator[None]:
    import scqubits.settings as scq

    old = scq.T1_DEFAULT_WARNING
    scq.T1_DEFAULT_WARNING = False
    try:
        yield
    finally:
        scq.T1_DEFAULT_WARNING = old


def _param_grid_values(value: ParamGridInput, precision: float) -> NDArray[np.float64]:
    if isinstance(value, tuple):
        return np.arange(value[0], value[1], precision)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    return np.array([float(value)], dtype=np.float64)


def generate_params_table(
    EJ: ParamGridInput,
    EC: ParamGridInput,
    EL: ParamGridInput,
    flux: float = 0.5,
    precision: float = 0.1,
) -> pd.DataFrame:
    """
    Create a table with columns: flux, EJ, EC, EL

    Args:
        EJ: EJ float or np.ndarray
        EC: EC float or np.ndarray
        EL: EL float or np.ndarray
        flux: flux value

    Returns:
        DataFrame with columns: flux, EJ, EC, EL
    """

    EJ = _param_grid_values(EJ, precision)
    EC = _param_grid_values(EC, precision)
    EL = _param_grid_values(EL, precision)

    return pd.DataFrame(
        [
            {
                "flux": flux,
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

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    # Iterate over raw numpy columns instead of pandas rows: the per-cell work
    # is a single scqubits eigensys call, so pandas' per-row Series overhead
    # dominated the runtime. Results are collected into an object array and
    # assigned once to keep the "esys" column dtype/content identical.
    fluxs = params_table["flux"].to_numpy()
    eJs = params_table["EJ"].to_numpy()
    eCs = params_table["EC"].to_numpy()
    eLs = params_table["EL"].to_numpy()

    esys_out = np.empty(len(params_table), dtype=object)
    for i in tqdm(range(len(params_table)), desc="Calculating esys"):
        fluxonium.flux = fluxs[i]
        fluxonium.EJ = eJs[i]
        fluxonium.EC = eCs[i]
        fluxonium.EL = eLs[i]
        esys_out[i] = fluxonium.eigensys(evals_count=fluxonium.truncated_dim)

    params_table["esys"] = esys_out


def calculate_f01(params_table: pd.DataFrame) -> None:
    """
    計算每個參數組合下的 f01, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "f01" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    esys = params_table["esys"].to_numpy()
    f01 = np.fromiter(
        (es[0][1] - es[0][0] for es in esys), dtype=np.float64, count=len(esys)
    )
    params_table["f01"] = f01


def calculate_m01(params_table: pd.DataFrame) -> None:
    """
    計算每個參數組合下的 m01, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "m01" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    fluxs = params_table["flux"].to_numpy()
    eJs = params_table["EJ"].to_numpy()
    eCs = params_table["EC"].to_numpy()
    eLs = params_table["EL"].to_numpy()
    esys = params_table["esys"].to_numpy()

    m01 = np.empty(len(params_table), dtype=np.float64)
    for i in range(len(params_table)):
        fluxonium.flux = fluxs[i]
        fluxonium.EJ = eJs[i]
        fluxonium.EC = eCs[i]
        fluxonium.EL = eLs[i]
        m01[i] = np.abs(fluxonium.n_operator(energy_esys=esys[i])[0, 1])

    params_table["m01"] = m01


def calculate_dispersive_shift(
    params_table: pd.DataFrame, g: float, r_f: float
) -> None:
    params_list = np.asarray(params_table.to_dict(orient="records"), dtype=object)

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    def update_fn(fluxonium: Fluxonium, row: dict[str, Any]) -> None:
        fluxonium.flux = row["flux"]
        fluxonium.EJ = row["EJ"]
        fluxonium.EC = row["EC"]
        fluxonium.EL = row["EL"]

    chis = calculate_chi_sweep(params_list, update_fn, g, r_f, progress=True)
    params_table["Chi"] = np.abs(chis[:, 1] - chis[:, 0])


def calculate_snr(
    params_table: pd.DataFrame, g: float, r_f: float, rf_w: float, max_photon: int
) -> None:
    """Compute the ge-SNR design metric per cell, only for ``valid==True`` rows.

    The Floquet ge-SNR is by far the most expensive stage, so it runs only on
    cells that survived the cheap ``avoid_*`` filters; rows with ``valid==False``
    get ``snr = NaN``. Call the ``avoid_*`` helpers *before* this function. The
    downstream plot hides invalid cells and ``annotate_best_point`` only ranks
    valid cells, so NaN on filtered-out rows does not affect the final selection.
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    def _calc_single_snr(row) -> float:
        _, snrs = calc_ge_snr(
            params=(row["EJ"], row["EC"], row["EL"]),
            flux=row["flux"],
            r_f=r_f,
            rf_w=rf_w,
            g=g,
            qub_dim=DESIGN_EVALS_COUNT,
            qub_cutoff=DESIGN_CUTOFF,
            max_photon=max_photon,
            esys=row["esys"],
        )
        return np.sort(snrs)[-3]

    valid_mask = params_table["valid"].to_numpy()
    snr = np.full(len(params_table), np.nan, dtype=np.float64)

    # Cell-level parallelism over valid rows: each cell is ~0.5s of Floquet ODE
    # work (no inner joblib in calc_ge_snr, so no nested oversubscription).
    # Invalid cells stay NaN in the prefilled output.
    valid_positions = np.flatnonzero(valid_mask)
    valid_snrs = Parallel(n_jobs=-1)(
        delayed(_calc_single_snr)(params_table.iloc[pos])
        for pos in tqdm(valid_positions, desc="Calculating snr")
    )
    snr[valid_positions] = np.asarray(valid_snrs, dtype=np.float64)

    params_table["snr"] = snr


def calculate_t1(
    params_table: pd.DataFrame,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
) -> None:
    """
    計算每個參數組合下的 t1, 需要已經計算過 esys的params_table

    會在 params_table 中新增一個 "t1" 欄位
    """

    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(
        1.0, 1.0, 1.0, flux=0.5, cutoff=DESIGN_CUTOFF, truncated_dim=DESIGN_EVALS_COUNT
    )

    fluxs = params_table["flux"].to_numpy()
    eJs = params_table["EJ"].to_numpy()
    eCs = params_table["EC"].to_numpy()
    eLs = params_table["EL"].to_numpy()
    esys = params_table["esys"].to_numpy()

    t1 = np.empty(len(params_table), dtype=np.float64)
    with _t1_default_warning_disabled():
        for i in range(len(params_table)):
            fluxonium.flux = fluxs[i]
            fluxonium.EJ = eJs[i]
            fluxonium.EC = eCs[i]
            fluxonium.EL = eLs[i]
            t1[i] = fluxonium.t1_effective(
                noise_channels=noise_channels,
                common_noise_options=dict(i=1, j=0, T=Temp),
                esys=esys[i],
            )

    params_table["t1"] = t1


def avoid_collision(
    params_table: pd.DataFrame, avoid_freqs: list[float], threshold: float = 0.3
) -> None:
    """
    計算每個參數組合下的 collision, 需要已經計算過 esys 的params_table

    會在 params_table 中新增一個 "collision" 欄位
    """

    # check if esys is calculated
    if "esys" not in params_table.columns:
        raise ValueError("This function requires esys to be calculated")

    freqs = np.array(avoid_freqs)[None, :]
    esys = params_table["esys"].to_numpy()

    collision = np.empty(len(params_table), dtype=bool)
    for i in range(len(params_table)):
        evals = esys[i][0]
        e0x = evals - evals[0]
        e1x = evals - evals[1]

        # 檢查能階差與避免頻率的差距是否小於閾值
        e0x_collision = np.min(np.abs(e0x[:, None] - freqs), axis=0) < threshold
        e1x_collision = np.min(np.abs(e1x[:, None] - freqs), axis=0) < threshold

        collision[i] = np.any(e0x_collision | e1x_collision)

    params_table["collision"] = collision
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
    params_table = params_table.copy()
    plot_table = params_table.drop(columns=["esys"], errors="ignore")

    # convert t1 from ns to us
    plot_table["t1"] *= 1e-3

    # Helper to build a descriptive label that conditionally includes additional flags
    def _build_label(row: pd.Series) -> str:
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
        x="snr",
        y="t1",
        color="valid",
        color_discrete_map={True: "blue", False: "red"},
        log_x=True,
        log_y=True,
        hover_name="Label",
        labels={"SNR": "snr", "T1 (us)": "t1"},
    )
    fig.update_traces(marker=dict(size=3))

    # 預設隱藏valid=False的點
    fig.for_each_trace(
        lambda trace: (
            trace.update(visible="legendonly") if trace.name == "False" else ()
        )
    )

    fig.update_layout(
        title_x=0.501,
        xaxis_title="SNR",
        yaxis_title="T1 (us)",
        template="plotly_white",
        showlegend=True,
        width=1100,
        height=750,
    )
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")

    return fig


def annotate_best_point(fig, data: pd.DataFrame) -> tuple[float, float, float]:
    """
    Find and plot the best snr's param on the plot, the equation of snr is:
        snr = snr * sqrt(T1)
    """

    # filter out invalid params
    valid_data = data[data["valid"]]

    snrs = valid_data["snr"] * np.sqrt(valid_data["t1"])
    best_param = valid_data.iloc[np.argmax(snrs)]

    EJ, EC, EL = (
        float(best_param["EJ"]),
        float(best_param["EC"]),
        float(best_param["EL"]),
    )

    fig.add_annotation(
        x=np.log10(best_param["snr"]),
        y=np.log10(best_param["t1"] * 1e-3),
        text=f"{EJ:.2f}/{EC:.2f}/{EL:.2f}",
        showarrow=True,
        arrowhead=1,
        arrowcolor="black",
        ax=0,
        ay=-20,
    )

    return EJ, EC, EL


def add_real_sample(
    fig: go.Figure,
    result_dir: str,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    flux: float = 0.5,
    rf_w: float = 7e-3,
    max_photon: int = 70,
) -> None:

    result_path = os.path.join(result_dir, "params.json")
    params_file = QubitParams(result_path, readonly=True)
    project = params_file.require_project()
    fit = params_file.require_fluxdep_fit()
    dispersive = params_file.get_dispersive_fit()
    assert dispersive is not None

    # unpack result
    name = project.name
    params = fit.params
    r_f = dispersive.bare_rf
    g = dispersive.g
    flux_half = fit.flux_half

    # load freq data
    sample_path = os.path.join(result_dir, "sample.csv")
    freq_df = pd.read_csv(sample_path)
    idx = np.argmin(np.abs(freq_df["calibrated mA"] - flux_half))
    t1 = freq_df["T1 (us)"].iloc[idx]

    # calculate chi
    _, snrs = calc_ge_snr(
        params=params,
        flux=flux,
        r_f=r_f,
        rf_w=rf_w,
        g=g,
        qub_dim=DESIGN_EVALS_COUNT,
        qub_cutoff=DESIGN_CUTOFF,
        max_photon=max_photon,
    )
    snr = np.sort(snrs)[-3]

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    # calculate t1
    fluxonium = Fluxonium(*params, flux=flux, cutoff=DESIGN_CUTOFF, truncated_dim=2)
    with _t1_default_warning_disabled():
        predict_t1 = 1e-3 * fluxonium.t1_effective(
            noise_channels=noise_channels,
            common_noise_options=dict(i=1, j=0, T=Temp),
        )

    # 添加從實際t1到預測t1的線段
    fig.add_shape(
        type="line",
        x0=snr,
        y0=t1,
        x1=snr,
        y1=predict_t1,
        line=dict(color="red", width=1, dash="dot"),
        name=name,
        legendgroup=name,
        showlegend=True,
    )

    # 添加實際t1的點
    fig.add_scatter(
        x=[snr],
        y=[t1],
        mode="markers+text",
        marker=dict(symbol="x", size=5, color="black"),
        text=[name],
        textposition="top center",
        hovertemplate=f"<b>{name}</b><br>"
        + f"SNR: {snr:.2f}<br>"
        + f"T1: {t1:.2f} us<br>"
        + f"EJ: {params[0]:.3f} GHz<br>"
        + f"EC: {params[1]:.3f} GHz<br>"
        + f"EL: {params[2]:.3f} GHz",
        legendgroup=name,
        showlegend=False,
    )

    # 添加預測t1的點
    fig.add_scatter(
        x=[snr],
        y=[predict_t1],
        mode="markers",
        marker=dict(symbol="x", size=5, color="red"),
        legendgroup=name,
        showlegend=False,
    )
