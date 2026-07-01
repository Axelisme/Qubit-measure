from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.utils.single_shot import GE_FitResult, singleshot_ge_analysis
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
)

from .util import classify_result, plot_with_classified

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------


def make_init_matrix(init_pops: NDArray[np.float64]) -> NDArray[np.float64]:
    p_gg_init = init_pops[0, 0]
    p_ge_init = init_pops[0, 1]
    # Symmetric preparation model: the pi pulse maps g<->e with the same leakage rate.
    p_eg_init = p_ge_init
    p_ee_init = p_gg_init
    p_go_init = 1.0 - p_gg_init - p_ge_init
    p_eo_init = p_go_init

    return np.array(
        [
            [p_gg_init, p_ge_init, p_go_init],
            [p_eg_init, p_ee_init, p_eo_init],
            [0.0, 0.0, 1.0],  # assume all initial to O
        ]
    )


def make_result_matrix(
    n_gg: float,
    n_ge: float,
    n_go: float,
    n_eg: float,
    n_ee: float,
    n_eo: float,
    n_og: float = 0.0,
    n_oe: float = 0.0,
    n_oo: float = 1.0,  # Other-state classification is modeled as perfectly isolated.
) -> NDArray[np.float64]:
    return np.array(
        [
            [n_gg, n_ge, n_go],
            [n_eg, n_ee, n_eo],
            [n_og, n_oe, n_oo],
        ]
    )


def solve_confusion_matrix(
    A_init: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    confusion_matrix = np.linalg.solve(A_init, Q)
    confusion_matrix = np.clip(confusion_matrix, 0.0, None)
    confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)

    return confusion_matrix


def calc_overlay(s: float, x: float, r: float) -> float:
    """計算二維高斯分佈在指定偏心圓內的比例"""
    from scipy.stats import ncx2

    if s <= 0:
        raise ValueError("標準差 s 必須大於 0")
    if r < 0:
        return 0.0

    # 自由度 k = 2 (對應二維空間)
    df = 2

    # 非中心參數 lambda = (dist_center / s)^2
    # 這裡圓心距離原點的距離為 x
    nc = (x / s) ** 2

    # 積分上限需歸一化： (r / s)^2
    limit = (r / s) ** 2

    # 使用非中心卡方分佈的累積分佈函數 (CDF)
    return ncx2.cdf(limit, df, nc).item()


def optimize_ge_radius(
    g_signals: NDArray[np.complex128],
    e_signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    init_pops: NDArray[np.float64],
    consider_other: bool = True,
) -> float:
    from scipy.optimize import minimize_scalar

    # use pca to retrive minimum radius
    ge_signals = np.concatenate([g_signals, e_signals])
    cov = np.cov(ge_signals.real, ge_signals.imag)
    eigenvalues, _ = np.linalg.eig(cov)
    sigma = np.sqrt(np.sort(eigenvalues)[0])  # minimum eigenvalue as sigma

    ge_dist = abs(g_center - e_center)
    A_init = make_init_matrix(init_pops)

    def loss_fn(radius: float) -> float:

        gg_mask, ge_mask, go_mask = classify_result(
            g_signals, g_center, e_center, radius
        )
        n_gg = gg_mask.sum() / gg_mask.shape[0]
        n_ge = ge_mask.sum() / ge_mask.shape[0]
        n_go = go_mask.sum() / go_mask.shape[0]

        eg_mask, ee_mask, eo_mask = classify_result(
            e_signals, g_center, e_center, radius
        )
        n_eg = eg_mask.sum() / eg_mask.shape[0]
        n_ee = ee_mask.sum() / ee_mask.shape[0]
        n_eo = eo_mask.sum() / eo_mask.shape[0]

        # assume other state is in the middle of g and e, calculate effective population
        n_og = calc_overlay(sigma, ge_dist / 2, radius) if consider_other else 0.0
        n_oe = n_og
        n_oo = 1.0 - n_og - n_oe

        Q = make_result_matrix(n_gg, n_ge, n_go, n_eg, n_ee, n_eo, n_og, n_oe, n_oo)
        confusion_matrix = solve_confusion_matrix(A_init, Q)

        # calculate condision number of confusion matrix as loss
        return np.linalg.cond(confusion_matrix)

    result = minimize_scalar(loss_fn, bounds=(0.0, ge_dist / 2))
    return float(result.x)  # type: ignore


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------


def _default_prepared_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class GE_Result:
    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]
    cfg_snapshot: GE_Cfg | None = None


class GEModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: PulseReadoutCfg


class GE_Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: GEModuleCfg
    shots: int


class GE_Exp(PersistableExperiment[GE_Result, GE_Cfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "shot_indices",
                "Shot Index",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis(
                "prepared_states",
                "Prepared State",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=GE_Result,
        cfg_type=GE_Cfg,
        tag="singleshot/ge",
    )

    @record_result
    def run(self, soc, soccfg, cfg: GE_Cfg) -> GE_Result:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)

        # Validate and setup configuration
        if cfg.rounds != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg.rounds = 1

        if cfg.reps != 1:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg.reps = cfg.shots

        def measure_fn(ctx: TaskState[NDArray[np.complex128], Any, GE_Cfg], _):
            modules = ctx.cfg.modules
            probe_cfg = None
            if ctx.env["with_probe"]:
                probe_cfg = modules.probe_pulse

            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", probe_cfg),
                    PulseReadout("readout", modules.readout),
                ],
            )
            prog.acquire(soc, progress=True, stop_checkers=[ctx.is_stop])

            acc_buf = prog.get_raw()
            assert acc_buf is not None

            length = cast(int, list(prog.ro_chs.values())[0]["length"])
            avgiq = acc_buf[0] / length  # (reps, 1, 2)

            return avgiq

        def raw2signal_fn(avgiq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, )
            signals = np.array(i0 + 1j * q0)  # (reps, )

            return signals

        with MeasureSession(cfg) as run:
            signals_buffer = run.buffer((2, cfg.shots), dtype=np.complex128)
            for step in run.scan("w/o probe pulse", [False, True]):
                run.env.update(with_probe=step.value)
                signals_buffer[step].measure(
                    measure_fn,
                    raw2signal_fn=raw2signal_fn,
                    pbar_n=1,
                )
            signals = signals_buffer.array

        return GE_Result(
            signals=signals,
            shot_indices=np.arange(signals.shape[1], dtype=np.int64),
            prepared_states=_default_prepared_states(),
            cfg_snapshot=cfg,
        )

    @retrieve_result
    def analyze(
        self,
        result: GE_Result | None = None,
        backend: Literal["center", "regression", "pca"] = "pca",
        **kwargs,
    ) -> tuple[float, NDArray[np.float64], GE_FitResult, Figure]:
        assert result is not None, "no result found"

        signals = result.signals

        return singleshot_ge_analysis(signals, backend=backend, **kwargs)

    @retrieve_result
    def calc_confusion_matrix(
        self,
        init_pops: NDArray[np.float64],
        g_center: complex,
        e_center: complex,
        radius: float | None = None,
        result: GE_Result | None = None,
        consider_other: bool = True,
    ) -> tuple[NDArray[np.float64], float, Figure]:
        assert result is not None, "no result found"

        signals = result.signals
        g_signals, e_signals = signals[0], signals[1]

        A_init = make_init_matrix(init_pops)

        if radius is None:
            radius = optimize_ge_radius(
                g_signals,
                e_signals,
                g_center,
                e_center,
                init_pops,
                consider_other=consider_other,
            )

        gg_mask, ge_mask, go_mask = classify_result(
            g_signals, g_center, e_center, radius
        )
        n_gg = gg_mask.sum() / gg_mask.shape[0]
        n_ge = ge_mask.sum() / ge_mask.shape[0]
        n_go = go_mask.sum() / go_mask.shape[0]

        eg_mask, ee_mask, eo_mask = classify_result(
            e_signals, g_center, e_center, radius
        )
        n_eg = eg_mask.sum() / eg_mask.shape[0]
        n_ee = ee_mask.sum() / ee_mask.shape[0]
        n_eo = eo_mask.sum() / eo_mask.shape[0]

        Q = make_result_matrix(n_gg, n_ge, n_go, n_eg, n_ee, n_eo)
        confusion_matrix = solve_confusion_matrix(A_init, Q)

        cond_number = np.linalg.cond(confusion_matrix)

        # plot confusion matrix
        fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))

        g_label = r"$|0\rangle$"
        e_label = r"$|1\rangle$"
        l_label = r"$|L\rangle$"

        plot_with_classified(ax1, g_signals, g_center, e_center, radius)
        plot_with_classified(ax2, e_signals, g_center, e_center, radius)

        ax1.set_title(
            f"{g_label}: {n_gg:.1%}, {e_label}: {n_ge:.1%}, {l_label}: {n_go:.1%}"
        )
        ax2.set_title(
            f"{g_label}: {n_eg:.1%}, {e_label}: {n_ee:.1%}, {l_label}: {n_eo:.1%}"
        )
        ax1.set_xlabel("")

        im = ax4.imshow(A_init, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax4)

        for i in range(A_init.shape[0]):
            for j in range(A_init.shape[1]):
                val = A_init[i, j]
                ax4.text(
                    j,
                    i,
                    f"{val:.2%}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels([g_label, e_label, l_label])
        ax4.set_yticklabels([g_label, e_label, l_label])
        ax4.set_xlabel("Actual State")
        ax4.set_ylabel("Prepared State")
        ax4.set_title("Initial Populations")

        im = ax3.imshow(confusion_matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax3)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                val = confusion_matrix[i, j]
                ax3.text(
                    j,
                    i,
                    f"{val:.2%}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels([g_label, e_label, l_label])
        ax3.set_yticklabels([g_label, e_label, l_label])
        ax3.set_xlabel("Measured State")
        ax3.set_ylabel("Actual State")
        ax3.set_title(f"Confusion Matrix (cond: {cond_number:.1f})")

        return confusion_matrix, radius, fig
