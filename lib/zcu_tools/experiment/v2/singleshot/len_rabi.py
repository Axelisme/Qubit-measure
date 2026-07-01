from __future__ import annotations

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
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.program.v2.twotone import TwoToneModuleCfg

from .util import calc_populations, correct_populations


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )
    cfg_snapshot: LenRabiCfg | None = None


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiCfg(TwoToneCfg, ExpCfgModel):
    modules: TwoToneModuleCfg
    sweep: LenRabiSweepCfg


class LenRabiExp(PersistableExperiment[LenRabiResult, LenRabiCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("lengths", "Length", "s", scale=US_TO_S, dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        tag="singleshot/len_rabi",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> LenRabiResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        length_param = sweep2param("length", cfg.sweep.length)
        modules.qub_pulse.set_param("length", length_param)

        with LivePlot1D(
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

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, LenRabiCfg], update_hook
            ):
                return TwoToneProgram(
                    soccfg,
                    ctx.cfg,
                    sweep=[("length", ctx.cfg.sweep.length)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    stop_checkers=[ctx.is_stop],
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )

            with MeasureSession(cfg) as run:
                buffer = run.buffer(
                    (len(lengths), 2),
                    dtype=np.float64,
                    on_update=lambda data: viewer.update(
                        lengths, calc_populations(data).T
                    ),
                )
                buffer.measure(
                    measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    pbar_n=1,
                )
                populations = buffer.array

        return LenRabiResult(lengths=lengths, signals=populations, cfg_snapshot=cfg)

    @retrieve_result
    def analyze(
        self,
        result: LenRabiResult | None = None,
        *,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> Figure:
        assert result is not None, "no result found"

        lens, populations = result.lengths, result.signals

        populations = calc_populations(populations)  # (len, geo)

        populations = correct_populations(populations, confusion_matrix)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        plot_kwargs: dict[str, Any] = dict(ls="-", marker="o", markersize=3)
        ax.plot(
            lens, populations[:, 0], color="blue", label="$|0\\rangle$", **plot_kwargs
        )
        ax.plot(
            lens, populations[:, 1], color="red", label="$|1\\rangle$", **plot_kwargs
        )
        ax.plot(
            lens, populations[:, 2], color="green", label="$|L\\rangle$", **plot_kwargs
        )
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Population (a.u.)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig
