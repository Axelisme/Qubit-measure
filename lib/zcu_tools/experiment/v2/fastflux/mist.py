from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
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
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
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
    cfg_snapshot: MistCfg | None = None


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

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: MistCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> MistResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
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

        with LivePlot2D("Flux Pulse Gain (a.u.)", "Mist Pulse Gain (a.u.)") as viewer:
            signals_buffer = SignalBuffer(
                (len(flux_gains), len(mist_gains)),
                on_update=lambda data: viewer.update(
                    flux_gains, mist_gains, mist_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
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
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return MistResult(flux_gains, mist_gains, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self, result: MistResult | None = None, ac_coeff: float | None = None
    ) -> Figure:
        assert result is not None, "No result found"

        flux_gains, mist_gains, signals2D = (
            result.flux_gains,
            result.mist_gains,
            result.signals,
        )

        real_signals = mist_signal2real(signals2D)

        fig, ax = plt.subplots(figsize=config.figsize)

        if ac_coeff is not None:
            mist_photons = ac_coeff * mist_gains**2
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

        return fig
