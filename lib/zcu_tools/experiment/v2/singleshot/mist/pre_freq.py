from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
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


class PreFreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    pi_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PreFreqCfg(ModularProgramCfg, TaskCfg):
    modules: PreFreqModuleCfg
    sweep: Dict[str, SweepCfg]


class PreFreqExp(AbsExperiment[PreFreqResult, PreFreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> PreFreqResult:
        _cfg = check_type(deepcopy(cfg), PreFreqCfg)  # prevent in-place modification
        modules = _cfg["modules"]

        _cfg["sweep"] = format_sweep1D(_cfg["sweep"], "freq")
        fpts = sweep2array(_cfg["sweep"]["freq"])  # predicted amplitudes

        Pulse.set_param(
            modules["init_pulse"], "freq", sweep2param("freq", _cfg["sweep"]["freq"])
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

            def measure_fn(ctx, update_hook):
                modules = ctx.cfg["modules"]
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset", {"type": "none"})),
                        Pulse(name="init_pulse", cfg=modules["init_pulse"]),
                        Pulse(name="pi_pulse", cfg=modules.get("pi_pulse")),
                        Pulse(name="probe_pulse", cfg=modules["probe_pulse"]),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(
                    soc,
                    progress=False,
                    callback=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    population_radius=radius,
                )

            signals = run_task(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(fpts), 2),
                    dtype=np.float64,
                ),
                init_cfg=_cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts, calc_populations(ctx.data).T
                ),
            )

        # record the last result
        self.last_cfg = _cfg
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
