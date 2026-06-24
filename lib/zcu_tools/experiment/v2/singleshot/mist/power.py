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
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
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

from ..util import calc_populations, correct_populations


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class PowerResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )
    cfg_snapshot: PowerCfg | None = None


class PowerModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    gain: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    sweep: PowerSweepCfg


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("gains", "Drive gain", "a.u.", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="singleshot/mist/power",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> PowerResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        gain_param = sweep2param("gain", cfg.sweep.gain)
        modules.probe_pulse.set_param("gain", gain_param)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, PowerCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.probe_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse(name="init_pulse", cfg=modules.init_pulse),
                    Pulse(name="probe_pulse", cfg=modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("gain", gain_sweep)],
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
            "Pulse gain",
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
                    result_shape=(len(gains), 2),
                    dtype=np.float64,
                    pbar_n=1,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, calc_populations(ctx.root_data).T
                ),
            )

        return PowerResult(gains=gains, signals=signals, cfg_snapshot=cfg)

    @retrieve_result
    def analyze(
        self,
        result: PowerResult | None = None,
        *,
        ac_coeff=None,
        log_scale=False,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> Figure:
        assert result is not None, "no result found"

        gains, populations = result.gains, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, confusion_matrix)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_kwargs = dict(ls="-", marker="o", markersize=1)
        ax.plot(xs, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(xs, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(xs, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
        if log_scale:
            ax.set_xscale("log")

        return fig
