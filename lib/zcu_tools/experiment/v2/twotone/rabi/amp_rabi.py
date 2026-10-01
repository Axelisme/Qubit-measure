from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import fit_rabi
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
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
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class AmpRabiResult:
    amps: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    skip: int = 0


@dataclass(frozen=True)
class AmpRabiAnalysis:
    pi_amp: float
    pi_amp_err: float
    pi2_amp: float
    pi2_amp_err: float


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AmpRabiSweepCfg(ConfigBase):
    gain: SweepCfg


class AmpRabiModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class AmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AmpRabiModuleCfg
    sweep: AmpRabiSweepCfg


class AmpRabiExp(PersistableExperiment[AmpRabiResult, AmpRabiCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("amps", "Gain", ""),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        tag="twotone/ge/rabi_gain",
    )

    def run(
        self,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        viewer = context.plots.liveplot_1d("measurement", "Pulse gain", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(gains),),
            on_update=lambda data: viewer.update(gains, rabi_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            cfg = sched.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            modules.qub_pulse.set_param("gain", sweep2param("gain", gain_sweep))

            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add_pulse("init_pulse", modules.init_pulse)
                .add_pulse("qubit_pulse", modules.qub_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("gain", gain_sweep)
                .build_and_acquire()
            )

        return AmpRabiResult(amps=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[AmpRabiCfg, AmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiAnalysis:
        result = source.result

        gains, signals = result.amps, result.signals
        gains = gains[options.skip :]
        signals = signals[options.skip :]

        real_signals = rabi_signal2real(signals)

        zero_signal = real_signals[np.argmin(np.abs(gains))]
        if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180.0

        pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, _, _, y_fit, _ = fit_rabi(
            gains, real_signals, decay=False, init_phase=init_phase
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(gains, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(gains, y_fit, label="fit")
        ax.axvline(
            pi_amp, ls="--", c="red", label=f"pi = {pi_amp:.3g} ± {pi_amp_err:.2g}"
        )
        ax.axvspan(pi_amp - pi_amp_err, pi_amp + pi_amp_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_amp,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_amp:.3g} ± {pi2_amp_err:.2g}",
        )
        ax.axvspan(pi2_amp - pi2_amp_err, pi2_amp + pi2_amp_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        # fit_rabi computes the per-gain fit uncertainties; surface them so the GUI
        # summary carries the gain errors (the figure labels already show them).
        return AmpRabiAnalysis(pi_amp, pi_amp_err, pi2_amp, pi2_amp_err)
