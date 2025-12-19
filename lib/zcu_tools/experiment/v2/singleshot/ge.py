from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, Optional, Tuple, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import make_ge_sweep
from zcu_tools.experiment.utils.single_shot import singleshot_ge_analysis
from zcu_tools.experiment.v2.runner import (
    HardTask,
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

# (signals)
GE_ResultType = NDArray[np.complex128]


class GE_TaskConfig(TaskConfig, TwoToneProgramCfg):
    probe_pulse: PulseCfg

    shots: int


class GE_Experiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: GE_TaskConfig) -> GE_ResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        # Validate and setup configuration
        if cfg.setdefault("rounds", 1) != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg["rounds"] = 1

        if "reps" in cfg:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

        if "sweep" in cfg:
            warnings.warn("sweep will be overwritten by singleshot measurement")

        # Create ge sweep: 0 (ground) and full gain (excited)
        cfg["sweep"] = {"ge": make_ge_sweep()}

        # Set qubit pulse gain from sweep parameter
        Pulse.set_param(
            cfg["probe_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        def measure_fn(ctx: TaskContextView, _):
            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                    Pulse("probe_pulse", ctx.cfg["probe_pulse"]),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            prog.acquire(soc, progress=True)

            acc_buf = prog.get_raw()
            assert acc_buf is not None

            length = cast(int, list(prog.ro_chs.values())[0]["length"])
            avgiq = acc_buf[0] / length  # (reps, *sweep, 1, 2)

            return avgiq

        def raw2signal_fn(avgiq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, *sweep)
            signals = np.array(i0 + 1j * q0)  # (reps, *sweep)

            # Swap axes to (*sweep, reps) format: (ge, shots)
            signals = np.swapaxes(signals, 0, -1)

            return signals

        signals = run_task(
            task=HardTask(
                measure_fn=measure_fn,
                raw2signal_fn=raw2signal_fn,
                result_shape=(2, cfg["shots"]),
            ),
            init_cfg=cfg,
        )

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
        init_populations: NDArray[np.float64],
        result: Optional[GE_ResultType] = None,
    ) -> Tuple[NDArray[np.float64], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        signals = result

        g_signals, e_signals = signals[0], signals[1]

        p_gg = np.mean(np.abs(g_signals - g_center) <= radius)
        p_ge = np.mean(np.abs(g_signals - e_center) <= radius)
        p_eg = np.mean(np.abs(e_signals - g_center) <= radius)
        p_ee = np.mean(np.abs(e_signals - e_center) <= radius)
        p_go = 1.0 - p_gg - p_ge
        p_eo = 1.0 - p_eg - p_ee

        Q = np.array(
            [
                [p_gg, p_ge, p_go],
                [p_eg, p_ee, p_eo],
            ]
        )

        C_ge = np.linalg.solve(init_populations, Q)

        C_ge[C_ge < 0] = 0.0  # avoid negative values due to numerical errors
        C_ge /= C_ge.sum(axis=1, keepdims=True)  # normalize

        # assume always correctly identify 'O' state
        confusion_matrix = np.concatenate([C_ge, np.array([[0.0, 0.0, 1.0]])], axis=0)

        # plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(confusion_matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                val = confusion_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2%}",
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["G", "E", "O"])
        ax.set_yticklabels(["G", "E", "O"])
        ax.set_xlabel("Measured State")
        ax.set_ylabel("Prepared State")
        ax.set_title("Confusion Matrix")

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
        self.last_result = signals

        return signals
