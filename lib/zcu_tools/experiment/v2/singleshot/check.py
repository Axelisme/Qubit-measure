from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired, Optional, cast, Union

from zcu_tools.experiment import AbsExperiment
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
    ReadoutCfg,
    Reset,
    ResetCfg,
    TwoToneProgramCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

# (signals)
CheckResultType = NDArray[np.complex128]


class CheckTaskConfig(TaskConfig, TwoToneProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    shots: int


class CheckExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: CheckTaskConfig) -> CheckResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        # Validate and setup configuration
        if cfg.setdefault("rounds", 1) != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg["rounds"] = 1

        if "reps" in cfg:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg["reps"] = cfg["shots"]

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
            avgiq = acc_buf[0] / length  # (reps, 1, 2)

            return avgiq

        def raw2signal_fn(avgiq: NDArray[np.float64]) -> NDArray[np.complex128]:
            i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]  # (reps, )
            return np.array(i0 + 1j * q0)  # (reps, )

        signals = run_task(
            task=HardTask(
                measure_fn=measure_fn,
                raw2signal_fn=raw2signal_fn,
                result_shape=(cfg["shots"],),
            ),
            init_cfg=cfg,
        )

        # Cache results
        self.last_cfg = cfg
        self.last_result = signals

        return signals

    def analyze(
        self,
        /,
        g_center: complex,
        e_center: complex,
        radius: float,
        result: Optional[CheckResultType] = None,
        numbins: Union[int, str] = "auto",
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        signals = result

        # Classify shots
        dists_g = np.abs(signals - g_center)
        dists_e = np.abs(signals - e_center)
        mask_g = dists_g < radius
        mask_e = dists_e < radius
        mask_o = ~(mask_g | mask_e)

        ng = np.sum(mask_g) / signals.shape[0]
        ne = np.sum(mask_e) / signals.shape[0]
        no = np.sum(mask_o) / signals.shape[0]

        fig, ax = plt.subplots(figsize=(6, 6))

        colors = np.full(signals.shape, "gray")
        colors[mask_g] = "blue"
        colors[mask_e] = "red"
        colors[mask_o] = "green"

        # plot shots
        ax.scatter(signals.real, signals.imag, c=colors, s=1, alpha=0.5)

        # plot centers with circle
        plt_params = dict(linestyle=":", color="k", marker="o", markersize=5)
        ax.plot(g_center.real, g_center.imag, markerfacecolor="b", **plt_params)
        ax.plot(e_center.real, e_center.imag, markerfacecolor="r", **plt_params)
        ax.add_patch(
            Circle(
                (g_center.real, g_center.imag),
                radius,
                color="b",
                fill=False,
                linestyle="--",
            )
        )
        ax.add_patch(
            Circle(
                (e_center.real, e_center.imag),
                radius,
                color="r",
                fill=False,
                linestyle="--",
            )
        )

        ax.set_xlabel("I value (a.u.)")
        ax.set_ylabel("Q value (a.u.)")

        ax.set_title(f"Population: G: {ng:.3f}, E: {ne:.3f}, O: {no:.3f}")

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[CheckResultType] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        signals = result

        shots = np.arange(signals.shape[0])

        save_data(
            filepath=filepath,
            x_info={"name": "shot", "unit": "point", "values": shots},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CheckResultType:
        signals, _, _ = load_data(filepath, **kwargs)
        signals = cast(CheckResultType, signals)

        self.last_cfg = None
        self.last_result = signals

        return signals
