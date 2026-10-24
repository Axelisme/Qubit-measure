from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Delay, Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune


@dataclass(frozen=True)
class T2EchoResult:
    """Total delays in us and mean complex IQ, one sample per delay.

    true_activate_detune is the phase-ramp frequency in MHz after hardware
    time/phase rounding, not a fitted qubit detuning. It is run-only metadata
    and is None after canonical load.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T2EchoCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune: float = 0.0


@dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    fit_method: Literal["fringe", "decay"] = "decay"
    fit_phase: bool = False


@dataclass(frozen=True)
class T2EchoAnalysis:
    t2e: float
    t2e_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


class T2EchoExp(Experiment[T2EchoResult, T2EchoCfg]):
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    # times stores us in memory, s on disk -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[T2EchoResult, T2EchoCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", "Time", "s", scale=1e-6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=T2EchoResult,
        cfg_type=T2EchoCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/t2echo",
    )

    def run(self, cfg: T2EchoCfg, *, context: RunContext) -> T2EchoResult:
        """Measure refocused coherence with a Hahn-echo delay sweep.

        After optional reset, apply pi/2, half the total delay, pi, the other
        half delay, then pi/2 and readout. cfg supplies the pulses and total
        delay sweep in us. detune in MHz ramps the final pi/2 phase with total
        delay to add fringes; it does not change the drive frequency.
        Return hardware-rounded total delays and mean complex IQ. analyze
        extracts T2echo from the decay or fringe envelope.
        """
        # 1. Round both half-delays and report the realized phase-ramp frequency.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune = cfg.detune

        lengths = sweep2array(cfg.length, "time", {"soccfg": soccfg, "scaler": 0.5})

        true_detune = actual_detune(
            cfg.length,
            detune=detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=0.5,
        )

        # 2. Display the IQ projection while averaging the total-delay sweep.
        viewer = live_line(
            context.plots,
            "measurement",
            "Time (us)",
            "Amplitude",
            title=f"T2 Echo (detune={true_detune:.3f}MHz)",
        )
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, t2echo_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            modules = cfg

            length_sweep = cfg.length
            length_param = sweep2param("length", length_sweep)
            detune_param = 360 * detune * length_param

            # 3. Refocus at the midpoint, then project with the programmed phase ramp.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("pi2_pulse1", modules.pi2_pulse),
                    Delay("t2e_delay1", delay=0.5 * length_param),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Delay("t2e_delay2", delay=0.5 * length_param),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=modules.pi2_pulse.with_updates(
                            phase=modules.pi2_pulse.phase + detune_param
                        ),
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("length", length_sweep)
                .build_and_acquire(rounds=cfg.rounds)
            )

        # Keep measured IQ separate from fitted coherence times and fringe frequency.
        return T2EchoResult(
            times=lengths,
            signals=signals_buffer.array,
            true_activate_detune=true_detune,
        )

    def analyze(
        self,
        source: RunRecord[T2EchoCfg, T2EchoResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2EchoAnalysis:
        """Fit T2echo in us from the IQ projection, excluding the first sample.

        fit_method selects a pure decay or a decaying fringe. fit_phase frees
        the fringe phase; otherwise it is fixed at zero. Decay-only fits ignore
        fit_phase and return zero detune. Fringe detune is the observed MHz
        frequency, including the applied phase ramp, not a corrected qubit
        detuning. Return fit uncertainties and quality, and draw the fit.
        Zero source detune forces decay. For fringe eligibility use cfg.detune
        when present, otherwise finite result.true_activate_detune. Missing
        detune for a requested fringe fit raises ValueError; explicit decay needs
        no detune source. Unknown fit_method values raise ValueError.
        """
        result = source.result
        fit_method = options.fit_method
        fit_phase = options.fit_phase
        if fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {fit_method}")
        if fit_method == "fringe":
            detune_source = (
                source.cfg.detune
                if source.cfg is not None
                else result.true_activate_detune
            )
            if detune_source is None or not np.isfinite(detune_source):
                raise ValueError(
                    "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                )
            if detune_source == 0.0:
                fit_method = "decay"

        xs, signals = result.times, result.signals

        xs = xs[1:]
        signals = signals[1:]

        # Project IQ onto its signal axis before choosing the envelope model.
        real_signals = rotate2real(signals).real

        if fit_method == "fringe":
            fixedparams = None if fit_phase else [None, None, None, 0.0, None]
            t2e, t2eerr, detune, detune_err, y_fit, (pOpt, pCov) = fit_decay_fringe(
                xs, real_signals, fixedparams=fixedparams
            )
        elif fit_method == "decay":
            t2e, t2eerr, y_fit, (pOpt, pCov) = fit_decay(xs, real_signals)
            detune = 0.0
            detune_err = 0.0
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        names = (
            ("y0", "yscale", "freq", "phase", "decay_time")
            if fit_method == "fringe"
            else ("y0", "yscale", "decay_time")
        )
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t2e_str = f"{t2e:.2f}us ± {t2eerr:.2f}us"
        if fit_method == "fringe":
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            title = r"$T_{2echo}$ fringe = " + f"{t2e_str}, detune = {detune_str}"

        elif fit_method == "decay":
            title = r"$T_{2echo}$ decay = " + f"{t2e_str}"

        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return T2EchoAnalysis(
            t2e=float(t2e),
            t2e_err=float(t2eerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": quality},
        )
