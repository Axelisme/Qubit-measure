from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
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
from zcu_tools.utils.datasaver import load_data, save_data

from ..util import calc_populations

PreFreqResult = Tuple[NDArray[np.float64], NDArray[np.float64]]


class PreFreqCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    pi_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PreFreqExp(AbsExperiment[PreFreqResult, PreFreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: PreFreqCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> PreFreqResult:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted amplitudes

        Pulse.set_param(
            cfg["init_pulse"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D(
            "Pre Pulse Frequency",
            "MIST",
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
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse(name="init_pulse", cfg=ctx.cfg["init_pulse"]),
                                Pulse(name="pi_pulse", cfg=ctx.cfg.get("pi_pulse")),
                                Pulse(name="probe_pulse", cfg=ctx.cfg["probe_pulse"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(fpts), 2),
                    dtype=np.float64,
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts, calc_populations(ctx.data).T
                ),
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result: PreFreqResult = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[PreFreqResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, populations = result

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_kwargs = dict(ls="-", marker="o", markersize=1)
        ax.plot(fpts, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(fpts, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(fpts, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("Frequency (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[PreFreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/mist/pdr",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, populations = result

        save_data(
            filepath=filepath,
            x_info={"name": "PrePulse frequency", "unit": "Hz", "values": 1e6 * fpts},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Population", "unit": "a.u.", "values": populations.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PreFreqResult:
        populations, fpts, _ = load_data(filepath, **kwargs)

        fpts = fpts / 1e6  # convert to MHz

        self.last_cfg = None
        self.last_result = (fpts, populations)

        return fpts, populations
