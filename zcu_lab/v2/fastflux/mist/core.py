from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, Experiment, ProgramExpCfg, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Join, Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class MistResult:
    flux_gains: NDArray[np.float64]
    mist_gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class MistModuleCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    init_pulse: Pulse.Cfg | None = None
    flux_pulse: Pulse.Cfg
    mist_pulse: Pulse.Cfg
    readout: Readout.Cfg


class MistSweepCfg(ConfigBase):
    flux_gain: SweepCfg
    mist_gain: SweepCfg


class MistCfg(ProgramExpCfg):
    modules: MistModuleCfg
    sweep: MistSweepCfg


@dataclass(frozen=True)
class MistAnalyzeOptions:
    ac_coeff: float | None = None


class MistExp(Experiment[MistResult, MistCfg]):
    Options: ClassVar[type[MistAnalyzeOptions]] = MistAnalyzeOptions

    # both axes are gains in a.u. (no MHz/us conversion) -> scale=IDENTITY.
    # inner-first: signals.shape == (len(flux_gains), len(mist_gains)) ==
    # reversed(axes) lengths, so mist_gains is the inner axis, flux_gains the outer.
    AXES_SPEC: ClassVar[AxesSpec[MistResult, MistCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("mist_gains", "Mist Pulse Gain", "a.u."),
            AxesSpec.Axis("flux_gains", "Flux Pulse Gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=MistResult,
        cfg_type=MistCfg,
        cfg_schema_version="2.0",
        tag="fastflux/mist",
    )

    def run(self, config: MistCfg, *, context: RunContext) -> MistResult:
        """Map the MIST readout response versus flux and MIST pulse gains.

        Reset and apply the initialization pulse, then start flux and MIST
        branches together. Read out after both finish. Return the two gain axes
        in a.u. and mean complex IQ shaped (flux_gain, mist_gain). This contrast
        map is not a calibrated state population or transition probability.
        """
        # Quantize both gain axes for their respective generators.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
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

        # Display the averaged response without classifying individual shots.
        viewer = context.plots.liveplot_2d(
            "measurement", "Flux Pulse Gain (a.u.)", "Mist Pulse Gain (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (len(flux_gains), len(mist_gains)),
            on_update=lambda data: viewer.update(
                flux_gains, mist_gains, mist_signal2real(data)
            ),
        )

        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules
            modules.flux_pulse.set_param(
                "gain", sweep2param("flux_gain", sched.cfg.sweep.flux_gain)
            )
            modules.mist_pulse.set_param(
                "gain", sweep2param("mist_gain", sched.cfg.sweep.mist_gain)
            )

            # Prepare once per shot, apply the two parallel stimuli, then read out.
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
        """Plot IQ contrast against flux gain and squared MIST gain.

        options.ac_coeff, when supplied, scales squared gain to a photon-number
        axis; otherwise the vertical coordinate remains gain squared.
        """
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
