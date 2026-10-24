from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2._support import live_heatmap


@dataclass(frozen=True)
class T1Result:
    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T1Cfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Reset.Cfg | None = None
    flux_pulse: Pulse.Cfg
    pi_pulse: Pulse.Cfg
    readout: Readout.Cfg
    gain: SweepCfg
    length: SweepCfg


@dataclass(frozen=True)
class T1Analysis:
    t1s: NDArray[np.float64]
    t1errs: NDArray[np.float64]


class T1Exp(Experiment[T1Result, T1Cfg]):
    # inner lengths stores memory us on disk (disk seconds) -> scale=1e-6;
    # outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("lengths", "Time", "s", scale=1e-6),
            AxesSpec.Axis("gains", "Flux Pulse Gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="3.0",
        tag="fastflux/t1",
    )

    def run(self, config: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure relaxation during a flux pulse at each programmed flux gain.

        Reset, excite with π, apply the gain/length-swept flux pulse, then read
        out. Return flux gains in a.u., pulse lengths in us and mean complex IQ
        shaped (gain, length). Each row is a decay trace; T1 is fitted later,
        not a directly measured population or a conversion from gain to flux.
        """
        # Resolve flux gain and duration coordinates on the hardware grid.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        gain_sweep = cfg.gain
        length_sweep = cfg.length

        lf_ch = modules.flux_pulse.ch
        gains = sweep2array(gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": lf_ch})
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg, "gen_ch": lf_ch})

        # Keep the relaxation traces grouped by flux gain.
        viewer = live_heatmap(
            context.plots, "measurement", "Flux Pulse Gain (a.u.)", "Time (us)"
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=lambda data: viewer.update(gains, lengths, t1_signal2real(data)),
        )

        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            modules.flux_pulse.set_param("gain", sweep2param("gain", sched.cfg.gain))
            modules.flux_pulse.set_param(
                "length", sweep2param("length", sched.cfg.length)
            )

            # Let the excited state evolve during the flux pulse before readout.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("flux_pulse", modules.flux_pulse),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("gain", sched.cfg.gain)
                .declare_sweep("length", sched.cfg.length)
                .build_and_acquire(rounds=cfg.rounds)
            )

        return T1Result(gains, lengths, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: None,
        *,
        plots: Plots,
    ) -> T1Analysis:
        """Fit decay at each flux gain, returning T1 and fit errors in us.

        First estimate a common baseline from accepted preliminary fits, then
        refit each row with that baseline fixed. Rejected final fits retain NaN
        T1 values; their initialized zero errors do not indicate a valid fit.
        """
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
