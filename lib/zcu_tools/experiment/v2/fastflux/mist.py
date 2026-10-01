from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Join,
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
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class MistResult:
    flux_gains: NDArray[np.float64]
    mist_gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class MistModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    flux_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistSweepCfg(ConfigBase):
    flux_gain: SweepCfg
    mist_gain: SweepCfg


class MistCfg(ProgramV2Cfg, ExpCfgModel):
    modules: MistModuleCfg
    sweep: MistSweepCfg


@dataclass(frozen=True)
class MistAnalyzeOptions:
    ac_coeff: float | None = None


class MistExp(PersistableExperiment[MistResult, MistCfg]):
    # both axes are gains in a.u. (no MHz/us conversion) -> scale=IDENTITY.
    # inner-first: signals.shape == (len(flux_gains), len(mist_gains)) ==
    # reversed(axes) lengths, so mist_gains is the inner axis, flux_gains the outer.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("mist_gains", "Mist Pulse Gain", "a.u.", scale=IDENTITY),
            Axis("flux_gains", "Flux Pulse Gain", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MistResult,
        cfg_type=MistCfg,
        tag="fastflux/mist",
    )

    def run(self, config: MistCfg, *, context: RunContext) -> MistResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
        modules = cfg.modules

        flux_gain_sweep = cfg.sweep.flux_gain
        mist_gain_sweep = cfg.sweep.mist_gain

        flux_gains = sweep2array(
            flux_gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.flux_pulse.ch},
        )
        mist_gains = sweep2array(
            mist_gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.mist_pulse.ch},
        )

        viewer = context.plots.liveplot_2d(
            "measurement", "Flux Pulse Gain (a.u.)", "Mist Pulse Gain (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (len(flux_gains), len(mist_gains)),
            on_update=lambda data: viewer.update(
                flux_gains, mist_gains, mist_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.flux_pulse.set_param(
                "gain", sweep2param("flux_gain", sched.cfg.sweep.flux_gain)
            )
            modules.mist_pulse.set_param(
                "gain", sweep2param("mist_gain", sched.cfg.sweep.mist_gain)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Join(
                        Pulse("flux_pulse", modules.flux_pulse),
                        Pulse("mist_pulse", modules.mist_pulse),
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("flux_gain", sched.cfg.sweep.flux_gain)
                .declare_sweep("mist_gain", sched.cfg.sweep.mist_gain)
                .build_and_acquire()
            )

        return MistResult(flux_gains, mist_gains, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[MistCfg, MistResult],
        options: MistAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result

        flux_gains, mist_gains, signals2D = (
            result.flux_gains,
            result.mist_gains,
            result.signals,
        )

        real_signals = mist_signal2real(signals2D)

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        if options.ac_coeff is not None:
            mist_photons = options.ac_coeff * mist_gains**2
            ylabel = "Photon Number (a.u.)"
        else:
            mist_photons = mist_gains**2
            ylabel = "Mist Pulse Gain^2 (a.u.)"

        im = NonUniformImage(ax, interpolation="nearest", cmap="RdBu_r")
        im.set_data(flux_gains, mist_photons, real_signals.T)
        im.set_extent(
            (flux_gains[0], flux_gains[-1], mist_photons[0], mist_photons[-1])
        )
        ax.add_artist(im)
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel(ylabel)

        fig.tight_layout()
