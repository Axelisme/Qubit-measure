from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_ge_decay
from zcu_tools.utils.process import rotate2real


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (ge, times)


# (times, signals)
T1GEResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class T1GETaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1GEExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: T1GETaskConfig) -> T1GEResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        cfg["sweep"] = {"ge": make_ge_sweep(), "length": cfg["sweep"]["length"]}

        ts = sweep2array(cfg["sweep"]["length"])

        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs={"title": "T1 relaxation", "num_lines": 2},
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Delay(
                                    name="t1_delay",
                                    delay=sweep2param(
                                        "length", ctx.cfg["sweep"]["length"]
                                    ),
                                ),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(2, len(ts)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, t1_signal2real(ctx.data)),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self, result: Optional[T1GEResultType] = None, *, share_t1: bool = True
    ) -> Tuple[float, float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, signals = result

        real_signals = rotate2real(signals).real
        g_signals, e_signals = real_signals

        (g_t1, g_t1err, g_fit_signals), (e_t1, e_t1err, e_fit_signals) = fit_ge_decay(
            times, g_signals, e_signals, share_t1=share_t1
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(times, g_signals, label="Ground", color="blue")
        ax.plot(times, e_signals, label="Excited", color="red")
        ax.plot(times, g_fit_signals, label="Ground fit", color="blue", ls="--")
        ax.plot(times, e_fit_signals, label="Excited fit", color="red", ls="--")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(
            f"T1g = {g_t1:.2f}us ± {g_t1err:.2f}us, T1e = {e_t1:.2f}us ± {e_t1err:.2f}us"
        )

        fig.tight_layout()

        return g_t1, g_t1err, e_t1, e_t1err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1GEResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
