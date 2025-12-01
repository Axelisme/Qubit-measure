from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import Pulse, TwoToneProgram, TwoToneProgramCfg, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LenRabiResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    g_pop, e_pop = signals[:, 0], signals[:, 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1).real


class LenRabiSingleShotTaskConfig(TaskConfig, TwoToneProgramCfg): ...


class LenRabiSingleShotExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiSingleShotTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> LenRabiResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        assert cfg["qub_pulse"]["waveform"]["style"] in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted

        Pulse.set_param(
            cfg["qub_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        with LivePlotter1D(
            "Length (us)",
            "Signal",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            viewer.get_ax().set_ylim(0.0, 1.0)

            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(lens), 2),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    lens, rabi_signal2real(ctx.data).T
                ),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(
        self, result: Optional[LenRabiResultType] = None, *, decay: bool = True
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        populations = rabi_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        plot_kwargs = dict(ls="-", marker="o", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LenRabiResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_length_singleshot",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
