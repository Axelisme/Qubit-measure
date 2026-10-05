from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer

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
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)

from zcu_lab.v2._support.singleshot.util import calc_populations
from zcu_lab.v2._support.singleshot.util import correct_populations
from zcu_lab.v2._support.singleshot.util import raw_population_signal


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class PreFreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


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
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class PreFreqAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class PreFreqExp(PersistableExperiment[PreFreqResult, PreFreqCfg]):
    Options: ClassVar[type[PreFreqAnalyzeOptions]] = PreFreqAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis(
                "freqs",
                "PrePulse frequency",
                "Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
            ),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=PreFreqResult,
        cfg_type=PreFreqCfg,
        tag="singleshot/mist/pre_freq",
    )

    def run(self, cfg: PreFreqCfg, *, context: RunContext) -> PreFreqResult:
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.init_pulse.ch},
        )

        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Pre Pulse Frequency",
            "MIST",
            num_lines=3,
            configure_axes=configure_axes,
        )
        buffer = SignalBuffer(
            (len(freqs), 2),
            dtype=np.float64,
            on_update=lambda data: viewer.update(freqs, calc_populations(data).T),
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            run_cfg = sched.cfg
            modules = run_cfg.modules
            freq_sweep = run_cfg.sweep.freq
            modules.init_pulse.set_param("freq", sweep2param("freq", freq_sweep))
            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add_pulse("init_pulse", modules.init_pulse)
                .add_pulse("pi_pulse", modules.pi_pulse)
                .add_pulse("probe_pulse", modules.probe_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire(
                    raw2signal_fn=raw_population_signal,
                    g_center=cfg.g_center,
                    e_center=cfg.e_center,
                    ge_radius=cfg.radius,
                )
            )
        signals = buffer.array

        return PreFreqResult(freqs=freqs, signals=signals)

    def analyze(
        self,
        source: RunRecord[PreFreqCfg, PreFreqResult],
        options: PreFreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result
        freqs, populations = result.freqs, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, options.confusion_matrix)

        _, ax = plots.subplots("fit", figsize=(6, 6))

        ax.plot(
            freqs,
            populations[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.set_xlabel("Frequency (MHz)", fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
