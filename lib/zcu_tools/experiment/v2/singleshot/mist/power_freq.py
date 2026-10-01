from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
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
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)

from ..util import calc_populations, correct_populations, raw_population_signal


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
    g_center: complex
    e_center: complex
    radius: float

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class FreqPowerAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


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
        cfg: FreqPowerCfg,
        *,
        context: RunContext,
    ) -> FreqPowerResult:
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
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

        viewers = tuple(
            context.plots.liveplot_2d(
                f"measurement_{state}",
                "gain (a.u.)",
                "freq (MHz)",
                title=state.capitalize(),
                uniform=False,
            )
            for state in ("ground", "excited", "other")
        )

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)
            for i, viewer in enumerate(viewers):
                viewer.update(gains, freqs, populations[..., i])

        buffer = SignalBuffer(
            (len(gains), len(freqs), 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            sched.cfg.modules.probe_pulse.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            for gain, step in sched.scan("gain", gains.tolist()):
                modules = step.cfg.modules
                modules.probe_pulse.set_param("gain", gain)
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("probe_pulse", modules.probe_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", step.cfg.sweep.freq)
                    .build_and_acquire(
                        raw2signal_fn=raw_population_signal,
                        g_center=cfg.g_center,
                        e_center=cfg.e_center,
                        ge_radius=cfg.radius,
                    )
                )
        signals = buffer.array

        return FreqPowerResult(gains=gains, freqs=freqs, signals=signals)

    def analyze(
        self,
        source: RunRecord[FreqPowerCfg, FreqPowerResult],
        options: FreqPowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result
        gains, freqs, populations = result.gains, result.freqs, result.signals

        populations = calc_populations(populations)

        populations = correct_populations(populations, options.confusion_matrix)

        fig, _ = plots.subplots("fit", nrows=3, ncols=1, figsize=(8, 10))
        ax_g, ax_e, ax_o = fig.axes

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
