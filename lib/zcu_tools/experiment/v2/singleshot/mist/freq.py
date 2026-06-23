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
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: FreqCfg | None = None


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
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
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
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
                stop_checkers=[ctx.is_stop],
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
        self.last_result = FreqResult(freqs=freqs, signals=signals, cfg_snapshot=cfg)

        return self.last_result

    def analyze(
        self,
        result: FreqResult | None = None,
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
        ax.set_xlabel("probe freq (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)

        return fig

    def save(
        self,
        filepath: str,
        result: FreqResult | None = None,
        comment: str | None = None,
        tag: str = "singleshot/mist/freq",
        **kwargs,
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
            z=(
                "Population",
                "a.u.",
                populations.T,
            ),  # (Ny=2, Nx=len(freqs)): inner (freqs) last
            axes=[
                ("Drive Freq", "Hz", 1e6 * freqs),  # inner axis (x)
                (
                    "GE population",
                    "a.u.",
                    np.asarray([0, 1]),
                ),  # outer axis (y), synthesized
            ],
            comment=comment,
            tags=tag,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        ld = load_labber_data(filepath)

        freqs = 1e-6 * np.asarray(ld.axes[0].values, dtype=np.float64)  # Hz to MHz
        # native load_labber_data does NOT flip axes: ld.z is (Ny=2, Nx=len(freqs))
        # restore (len(freqs), 2) shape that result.signals expects
        populations = np.real(np.asarray(ld.z)).astype(np.float64).T
        comment = ld.comment

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = FreqResult(
            freqs=freqs, signals=populations, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
