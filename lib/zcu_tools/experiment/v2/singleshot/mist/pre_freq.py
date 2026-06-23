from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
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
from zcu_tools.utils.datasaver import safe_labber_filepath
from zcu_tools.utils.labber_io import load_labber_data, save_labber_data

from ..util import calc_populations, correct_populations


@dataclass(frozen=True)
class PreFreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: PreFreqCfg | None = None


class PreFreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg
    pi_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PreFreqSweepCfg(ConfigBase):
    freq: SweepCfg


class PreFreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PreFreqModuleCfg
    sweep: PreFreqSweepCfg


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
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.init_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, PreFreqCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.init_pulse.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )

        with LivePlot1D(
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
        self.last_result = PreFreqResult(freqs=freqs, signals=signals, cfg_snapshot=cfg)

        return self.last_result

    def analyze(
        self,
        result: PreFreqResult | None = None,
        *,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, populations = result.freqs, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, confusion_matrix)

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_kwargs = dict(ls="-", marker="o", markersize=1)
        ax.plot(freqs, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(freqs, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(freqs, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("Frequency (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)

        return fig

    def save(
        self,
        filepath: str,
        result: PreFreqResult | None = None,
        comment: str | None = None,
        tag: str = "singleshot/mist/pre_freq",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, populations = result.freqs, result.signals

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("result.cfg_snapshot is None")
        comment = make_comment(cfg, comment)

        save_labber_data(
            safe_labber_filepath(filepath),
            z=("Population", "a.u.", populations.T),  # native (Ny=2, Nx=len(freqs))
            axes=[
                ("PrePulse frequency", "Hz", 1e6 * freqs),  # inner-first: freq (Nx)
                ("GE population", "a.u.", np.array([0, 1])),  # outer: ge (Ny=2)
            ],
            comment=comment,
            tags=tag,
        )

    def load(self, filepath: str) -> PreFreqResult:
        data = load_labber_data(filepath)

        # native load returns z as (Ny=2, Nx=len(freqs)); transpose to (Nx, Ny)
        populations = np.real(data.z.T).astype(np.float64)
        freqs = data.axes[0].values / 1e6  # axes[0]=freq inner, Hz->MHz
        comment = data.comment

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = PreFreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = PreFreqResult(
            freqs=freqs, signals=populations, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
