from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import make_ge_sweep
from zcu_tools.experiment.utils.single_shot import singleshot_ge_analysis
from zcu_tools.experiment.v2.runner import (
    HardTask,
    SoftTask,
    TaskConfig,
    TaskContextView,
    run_task,
)
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    Reset,
    TwoToneProgramCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

from .util import plot_classified_result

# (signals)
GE_ResultType = NDArray[np.complex128]


class GE_TaskConfig(TaskConfig, TwoToneProgramCfg):
    probe_pulse: PulseCfg

    shots: int


class GE_Exp(AbsExperiment[GE_ResultType, GE_TaskConfig]):
    def run(self, soc, soccfg, cfg: GE_TaskConfig) -> GE_ResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        # Validate and setup configuration
        if cfg.setdefault("rounds", 1) != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg["rounds"] = 1

        if "reps" in cfg:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

        def measure_fn(ctx: TaskContextView, _):
            probe_cfg = None
            if ctx.env_dict["with_probe"]:
                probe_cfg = ctx.cfg["probe_pulse"]

            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                    Pulse("probe_pulse", probe_cfg),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            prog.acquire(soc, progress=True)

            acc_buf = prog.get_raw()
            assert acc_buf is not None

            length = cast(int, list(prog.ro_chs.values())[0]["length"])
            avgiq = acc_buf[0] / length  # (reps, 1, 2)

            return avgiq

        def raw2signal_fn(avgiq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, )
            signals = np.array(i0 + 1j * q0)  # (reps, )

            return signals

        signals = run_task(
            task=SoftTask(
                sweep_name="w/o probe pulse",
                sweep_values=[False, True],
                update_cfg_fn=lambda _, ctx, with_probe: ctx.env_dict.update(
                    with_probe=with_probe
                ),
                sub_task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=raw2signal_fn,
                    result_shape=(cfg["shots"],),
                ),
            ),
            init_cfg=cfg,
        )
        signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = signals

        return signals

    def analyze(
        self,
        result: Optional[GE_ResultType] = None,
        backend: Literal["center", "regression", "pca"] = "pca",
        **kwargs,
    ) -> Tuple[float, np.ndarray, dict, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        signals = result

        return singleshot_ge_analysis(signals, backend=backend, **kwargs)

    def calc_confusion_matrix(
        self,
        g_center: complex,
        e_center: complex,
        radius: float,
        init_pops: NDArray[np.float64],
        result: Optional[GE_ResultType] = None,
    ) -> Tuple[NDArray[np.float64], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        signals = result
        g_signals, e_signals = signals[0], signals[1]

        # 1. 取得初始化在 g, e 的群體 (2x2)
        # init_populations = [[p_gg, p_ge], [p_eg, p_ee]]
        p_gg_init = init_pops[0, 0]
        p_ge_init = init_pops[0, 1]
        p_eg_init = p_ge_init  # TODO: before fix singleshot fitting problem, assume use perfect pi pulse
        p_ee_init = p_gg_init
        p_go_init = 1.0 - p_gg_init - p_ge_init
        p_eo_init = p_go_init

        # 2. 構建完整的 3x3 初始化矩陣 A
        A_init = np.array(
            [
                [p_gg_init, p_ge_init, p_go_init],
                [p_eg_init, p_ee_init, p_eo_init],
                [0.0, 0.0, 1.0],  # assume all initial to O
            ]
        )

        # plot confusion matrix
        fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))

        n_gg, n_ge, n_go = plot_classified_result(
            ax1, g_signals, g_center, e_center, radius
        )
        n_eg, n_ee, n_eo = plot_classified_result(
            ax2, e_signals, g_center, e_center, radius
        )
        ax1.set_title(f"G: {n_gg:.1%}, E: {n_ge:.1%}, O: {n_go:.1%}")
        ax2.set_title(f"G: {n_eg:.1%}, E: {n_ee:.1%}, O: {n_eo:.1%}")

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
        ax4.set_xticklabels(["G", "E", "O"])
        ax4.set_yticklabels(["G", "E", "O"])
        ax4.set_xlabel("Actual State")
        ax4.set_ylabel("Prepared State")
        ax4.set_title("Initial Populations")

        Q = np.array(
            [
                [n_gg, n_ge, n_go],
                [n_eg, n_ee, n_eo],
                [0.0, 0.0, 1.0],  # assume all initial to O
            ]
        )

        # 4. 求解 C (此時 A 是 3x3 方陣，可以使用 solve)
        # 方程式：Q = A @ C  => C 代表「真實狀態轉移到測量狀態」的機率
        confusion_matrix = np.linalg.solve(A_init, Q)

        # 5. 數值修正與歸一化
        confusion_matrix[confusion_matrix < 0] = 0.0
        confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)

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
        ax3.set_xticklabels(["G", "E", "O"])
        ax3.set_yticklabels(["G", "E", "O"])
        ax3.set_xlabel("Measured State")
        ax3.set_ylabel("Actual State")
        ax3.set_title("Confusion Matrix")

        return confusion_matrix, fig

    def save(
        self,
        filepath: str,
        result: Optional[GE_ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/ge",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals = result

        save_data(
            filepath=filepath,
            x_info={
                "name": "shot",
                "unit": "point",
                "values": np.arange(signals.shape[1]),
            },
            y_info={"name": "ge", "unit": "", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> GE_ResultType:
        signals, _, _ = load_data(filepath, **kwargs)

        self.last_cfg = None
        self.last_result = signals.T

        return signals
