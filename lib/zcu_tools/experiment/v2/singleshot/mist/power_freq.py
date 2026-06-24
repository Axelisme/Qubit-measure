from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D, MultiLivePlot, make_plot_frame
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

from ..util import calc_populations, correct_populations


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class FreqPowerResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )
    cfg_snapshot: FreqPowerCfg | None = None


class FreqPowerModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class FreqPowerSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class FreqPowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqPowerModuleCfg
    sweep: FreqPowerSweepCfg


class FreqPowerExp(PersistableExperiment[FreqPowerResult, FreqPowerCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("freqs", "Drive Freq", "Hz", scale=MHZ_TO_HZ, dtype=np.float64),
            Axis("gains", "Drive gain", "a.u.", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=FreqPowerResult,
        cfg_type=FreqPowerCfg,
        tag="singleshot/mist/power_freq",
    )

    def run(
        self,
        soc,
        soccfg,
        cfg: FreqPowerCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> FreqPowerResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FreqPowerCfg],
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

        fig, axs = make_plot_frame(3, 1, plot_instant=True, figsize=(12, 6))

        with MultiLivePlot(
            fig,
            dict(
                plot_2d_g=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[0][0]]],
                ),
                plot_2d_e=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[1][0]]],
                ),
                plot_2d_o=LivePlot2D(
                    "gain (a.u.)",
                    "freq (MHz)",
                    uniform=False,
                    existed_axes=[[axs[1][1]]],
                ),
            ),
        ) as viewer:

            def plot_fn(ctx) -> None:
                populations = calc_populations(np.asarray(ctx.root_data))

                viewer.get_plotter("plot_2d_g").update(
                    gains, freqs, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    gains, freqs, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    gains, freqs, populations[..., 2], refresh=False
                )

                viewer.refresh()

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(freqs), 2),
                    dtype=np.float64,
                    pbar_n=1,
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda i, ctx, gain: (
                        ctx.cfg.modules.probe_pulse.set_param("gain", gain)
                    ),
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_result = FreqPowerResult(
            gains=gains, freqs=freqs, signals=signals, cfg_snapshot=cfg
        )

        return self.last_result

    def analyze(
        self,
        result: FreqPowerResult | None = None,
        *,
        ac_coeff=None,
        log_scale=False,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, populations = result.gains, result.freqs, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, confusion_matrix)

        fig, (ax_g, ax_e, ax_o) = plt.subplots(3, 1, figsize=(8, 10))

        im_g = ax_g.imshow(
            populations[..., 0],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_g.set_title("Ground State Population")
        ax_g.set_xlabel("Drive Gain (a.u.)")
        ax_g.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_g, ax=ax_g, label="Population (a.u.)")

        im_e = ax_e.imshow(
            populations[..., 1],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_e.set_title("Excited State Population")
        ax_e.set_xlabel("Drive Gain (a.u.)")
        ax_e.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_e, ax=ax_e, label="Population (a.u.)")

        im_o = ax_o.imshow(
            populations[..., 2],
            extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
            aspect="auto",
            origin="lower",
        )
        ax_o.set_title("Other States Population")
        ax_o.set_xlabel("Drive Gain (a.u.)")
        ax_o.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_o, ax=ax_o, label="Population (a.u.)")

        fig.tight_layout()

        return fig
