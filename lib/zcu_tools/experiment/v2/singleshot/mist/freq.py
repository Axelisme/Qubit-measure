from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import (
    Any,
    Callable,
    Optional,
    TypeAlias,
)

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

from ..util import calc_populations

FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class FreqModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


class FreqDepExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> FreqResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FreqCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.probe_pulse.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        with LivePlot1D(
            "Pulse freq",
            "Population",
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
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(freqs), 2),
                    dtype=np.float64,
                    pbar_n=1,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, calc_populations(ctx.root_data).T
                ),
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, populations = result

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_kwargs = dict(ls="-", marker="o", markersize=1)
        ax.plot(freqs, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(freqs, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(freqs, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("probe freq (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/mist/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, populations = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Freq", "unit": "Hz", "values": 1e6 * freqs},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Population", "unit": "a.u.", "values": populations.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        populations, freqs, _, comment = load_data(filepath, return_comment=True, **kwargs)

        freqs = 1e-6 * freqs  # Hz to MHz
        populations = np.real(populations).astype(np.float64)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (freqs, populations)

        return freqs, populations
