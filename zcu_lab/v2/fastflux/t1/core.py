from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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
class T1Result:
    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T1ModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    flux_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    gain: SweepCfg
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


@dataclass(frozen=True)
class T1Analysis:
    t1s: NDArray[np.float64]
    t1errs: NDArray[np.float64]


class T1Exp(PersistableExperiment[T1Result, T1Cfg]):
    # inner lengths stores memory us on disk (disk seconds) -> scale=US_TO_S;
    # outer gains -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("lengths", "Time", "s", scale=US_TO_S),
            Axis("gains", "Flux Pulse Gain", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        tag="fastflux/t1",
    )

    def run(self, config: T1Cfg, *, context: RunContext) -> T1Result:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain
        length_sweep = cfg.sweep.length

        # uniform in square space
        lf_ch = modules.flux_pulse.ch
        gains = sweep2array(gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": lf_ch})
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg, "gen_ch": lf_ch})

        viewer = context.plots.liveplot_2d(
            "measurement", "Flux Pulse Gain (a.u.)", "Time (us)"
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=lambda data: viewer.update(gains, lengths, t1_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.flux_pulse.set_param(
                "gain", sweep2param("gain", sched.cfg.sweep.gain)
            )
            modules.flux_pulse.set_param(
                "length", sweep2param("length", sched.cfg.sweep.length)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("flux_pulse", modules.flux_pulse),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .declare_sweep("length", sched.cfg.sweep.length)
                .build_and_acquire()
            )

        return T1Result(gains, lengths, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: None,
        *,
        plots: Plots,
    ) -> T1Analysis:
        del options  # This analysis has no configurable options.
        result = source.result

        gains, lengths, signals2D = result.gains, result.lengths, result.signals

        real_signals = t1_signal2real(signals2D)

        def is_good_fit(fit_signal, real_signal) -> bool:
            real_ptp = np.ptp(real_signal)
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            return fit_ptp > 0.5 * real_ptp and fit_ptp > np.mean(
                np.abs(smooth_residual)
            )

        prev_pOpt = None
        list_pOpt = []
        for real_signal in real_signals:
            *_, fit_signal, (pOpt, _) = fit_decay(
                lengths,
                real_signal,
                fit_params=prev_pOpt,
            )
            prev_pOpt = pOpt if is_good_fit(fit_signal, real_signal) else None
            list_pOpt.append(prev_pOpt)
        mean_y0 = np.median([pOpt[0] for pOpt in list_pOpt if pOpt is not None]).item()

        t1s = np.full_like(gains, np.nan)
        t1errs = np.zeros_like(gains)
        for i, real_signal in enumerate(real_signals):
            t1, t1err, fit_signal, *_ = fit_decay(
                lengths,
                real_signals[i, :],
                fit_params=list_pOpt[i],
                fixedparams=(mean_y0, None, None),
            )
            if is_good_fit(fit_signal, real_signal):
                t1s[i] = t1
                t1errs[i] = t1err

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            real_signals.T,
            extent=(
                float(gains[0]),
                float(gains[-1]),
                float(lengths[0]),
                float(lengths[-1]),
            ),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_ylabel("Time (us)")

        ax.errorbar(gains, t1s, yerr=t1errs, fmt=".", label="T1", color="black")
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.legend()

        fig.tight_layout()

        return T1Analysis(t1s=t1s, t1errs=t1errs)
