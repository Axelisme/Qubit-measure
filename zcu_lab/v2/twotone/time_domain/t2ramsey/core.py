from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import (
    FitQuality,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Delay, Pulse, Readout, Reset, SweepCfg, TimelineBuilder
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


@dataclass(frozen=True)
class T2RamseyResult:
    """Free-evolution delays in us and mean complex IQ, one sample per delay.

    true_activate_detune is the phase-ramp frequency in MHz after hardware
    time/phase rounding, not a fitted qubit detuning. It is run-only metadata
    and is None after canonical load.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None


class T2RamseyCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune: float = 0.0
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop averaging after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


@validated_dataclass(frozen=True)
class T2RamseyAnalyzeOptions:
    fit_fringe: Annotated[bool, FieldMeta(label="Fit fringe")] = True
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset (fringe only)")] = True


@dataclass(frozen=True)
class T2RamseyAnalysis:
    t2r: float
    t2r_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


class T2RamseyExp(Experiment[T2RamseyResult, T2RamseyCfg]):
    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions

    # times stored as seconds on disk -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[T2RamseyResult, T2RamseyCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", label="Time", unit="s", scale=1e-6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=T2RamseyResult,
        cfg_type=T2RamseyCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t2ramsey",
    )

    def run(self, cfg: T2RamseyCfg, *, context: RunContext) -> T2RamseyResult:
        """Measure Ramsey fringes and free-evolution coherence decay.

        After optional reset, apply pi/2, wait the swept delay, then apply the
        final pi/2 and read out. cfg supplies the pulses and delay sweep in us.
        detune in MHz ramps the final pi/2 phase with delay to add fringes;
        it does not change the drive frequency. Return hardware-rounded delays
        and mean complex IQ. analyze fits the envelope to estimate T2 Ramsey
        and, for a fringe fit, the observed oscillation frequency.
        """
        # 1. Prepare the delay grid and the realized phase-ramp frequency.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune = cfg.detune

        length_sweep = cfg.delay
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})

        true_detune = actual_detune(
            cfg.delay,
            detune=detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        )

        # 2. Display the IQ projection while averaging the delay sweep.
        title = f"T2 Ramsey - detune {detune:.2f} MHz"
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Time (us)", ylabel="Real Signal")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, t2ramsey_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=update,
        )
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length_sweep = cfg.delay
            length_param = builder.add_sweep("delay", length_sweep)
            detune_param = 360 * detune * length_param

            # 3. Let coherence evolve freely, then project with the phase ramp.

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("pi2_pulse1", cfg.pi2_pulse))
            builder.emit(Delay("t2_delay", delay=length_param))
            builder.delay_by_timeline()
            builder.emit(
                Pulse(
                    name="pi2_pulse2",
                    cfg=cfg.pi2_pulse.with_updates(
                        phase=cfg.pi2_pulse.phase + detune_param
                    ),
                )
            )
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                soc,
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals_buffer.at(), cfg.earlystop_snr, t2ramsey_signal2real
                    ),
                ),
            )

        # Keep measured IQ separate from fitted coherence times and fringe frequency.
        return T2RamseyResult(
            times=lengths,
            signals=signals_buffer.array,
            true_activate_detune=true_detune,
        )

    def analyze(
        self,
        source: RunRecord[T2RamseyCfg, T2RamseyResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2RamseyAnalysis:
        """Fit T2 Ramsey in us and optionally the observed fringe frequency.

        Project source IQ onto its signal axis. fit_fringe selects a decaying
        fringe instead of a pure decay. fit_phase frees the fringe phase;
        otherwise the near-zero-delay signal selects a fixed 0/180 degree
        phase. Decay-only fits ignore fit_phase and return zero detune.
        Fringe detune is in MHz and includes the applied phase ramp, not a
        corrected qubit detuning. Return uncertainties and quality, and draw
        the fit using the same samples.
        """
        result = source.result
        fit_fringe = options.fit_fringe
        fit_phase = options.fit_phase

        lengths, signals = result.times, result.signals

        real_signals = t2ramsey_signal2real(signals)

        # Choose whether to fit oscillations as well as their decay envelope.
        if fit_fringe:
            fixedparams: list[float | None] | None = None
            if not fit_phase:
                zero_signal = real_signals[np.argmin(np.abs(lengths))]
                if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
                    init_phase = 0.0
                else:
                    init_phase = 180.0
                fixedparams = [None, None, None, init_phase, None]
            fit = fit_decay_fringe(lengths, real_signals, fixedparams=fixedparams)
            t2r, t2rerr, detune, detune_err, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.freq,
                fit.freq_err,
                fit.curve,
            )
        else:
            fit = fit_decay(lengths, real_signals)
            t2r, t2rerr, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.curve,
            )
            detune = 0.0
            detune_err = 0.0

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(lengths, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lengths, y_fit, label="fit")
        t2r_str = f"{t2r:.2f}us ± {t2rerr:.2f}us"
        if fit_fringe:
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            ax.set_title(f"T2 fringe = {t2r_str}, detune = {detune_str}", fontsize=15)
        else:
            ax.set_title(f"T2 decay = {t2r_str}", fontsize=15)
        ax.set_xlabel("Delay Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return T2RamseyAnalysis(
            t2r=float(t2r),
            t2r_err=float(t2rerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": fit.quality},
        )
