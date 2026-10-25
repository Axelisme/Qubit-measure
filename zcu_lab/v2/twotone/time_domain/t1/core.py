from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import zcu_tools.analysis.fitting as ft
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_dual_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class T1Result:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1Cfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg | list[float], FieldMeta(label="Delay", unit="us")]
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = True


@dataclass(frozen=True)
class T1AnalyzeOptions:
    dual_exp: Annotated[bool, FieldMeta(label="Dual exponential")] = False
    skip: Annotated[int, FieldMeta(label="Skip leading points")] = 0


@dataclass(frozen=True)
class T1Analysis:
    t1: float
    t1_err: float
    t1b: float | None = None
    t1b_err: float | None = None
    fit_quality: Mapping[str, FitQuality] | None = None


class T1Exp(Experiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions

    # times stored as seconds on disk -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", "Time", "s", 1e-6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/t1",
    )

    def run(self, config: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure energy relaxation with reset → π → delay → readout.

        config selects a linear delay sweep or nonuniform delays in us.
        Uniform sampling requires SweepCfg; nonuniform delays must remain
        distinct and increasing after hardware quantization or raise ValueError.
        context supplies acquisition and live plots. Return quantized delays
        and mean complex IQ with shape (time,); unmeasured points remain NaN.
        """
        # 1. Prepare the delay axis without changing the caller's configuration.
        cfg = deepcopy(config)
        if cfg.uniform and not isinstance(cfg.length, SweepCfg):
            raise ValueError("uniform mode requires SweepCfg")
        soc, soccfg, plots = context.soc, context.soccfg, context.plots

        if cfg.uniform:
            lengths = sweep2array(cfg.length, "time", {"soccfg": soccfg})
            viewer = live_line(
                plots, "measurement", "Time (us)", "Amplitude", title="T1 relaxation"
            )
            signals_buffer = SignalBuffer(
                (len(lengths),),
                on_update=lambda data: viewer.update(lengths, t1_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
                cfg = sched.cfg
                length_sweep = cfg.length
                if not isinstance(length_sweep, SweepCfg):
                    raise TypeError("uniform mode requires SweepCfg")
                length_param = sweep2param("length", length_sweep)

                # 2a. Excite, wait a linearly swept delay, then acquire readout IQ.
                _ = (
                    sched.prog_builder(
                        soc,
                        soccfg,
                        reps=sched.cfg.reps,
                        relax_delay=sched.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", cfg.reset),
                        Pulse("pi_pulse", cfg.pi_pulse),
                        Delay("t1_delay", length_param),
                        Readout("readout", cfg.readout),
                    )
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(
                        rounds=cfg.rounds,
                        options=AcquireOptions(
                            retry=cfg.acquire_retry,
                            stop_condition=snr_checker(
                                signals_buffer.at(), cfg.earlystop_snr, t1_signal2real
                            ),
                        ),
                    )
                )
        else:
            delay_table = materialize_nonuniform_t1_delays(
                cfg.length,
                soccfg=soccfg,
            )
            length_cycles = delay_table.cycles
            lengths = delay_table.times_us
            viewer = live_line(plots, "measurement", "Time (us)", "Amplitude")
            signals_buffer = SignalBuffer(
                (len(lengths),),
                on_update=lambda data: viewer.update(lengths, t1_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
                cfg = sched.cfg

                # 2b. Load each nonuniform delay in cycles before reset and excitation.
                _ = (
                    sched.prog_builder(
                        soc,
                        soccfg,
                        reps=sched.cfg.reps,
                        relax_delay=sched.cfg.relax_delay,
                    )
                    .add(
                        LoadValue(
                            "load_t1_delay",
                            values=list(length_cycles),
                            idx_reg="length_idx",
                            val_reg="t1_delay_cycle",
                            auto_compress=False,
                        ),
                        Reset("reset", cfg.reset),
                        Pulse("pi_pulse", cfg.pi_pulse),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", cfg.readout),
                    )
                    .declare_sweep("length_idx", len(length_cycles))
                    .build_and_acquire(
                        rounds=cfg.rounds,
                        options=AcquireOptions(
                            retry=cfg.acquire_retry,
                            stop_condition=snr_checker(
                                signals_buffer.at(), cfg.earlystop_snr, t1_signal2real
                            ),
                        ),
                    )
                )

        # 3. Keep acquired averages and any NaN partial-data slots on the same axis.
        return T1Result(times=lengths, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1Analysis:
        result = source.result
        xs = result.times[options.skip :]
        signals = result.signals[options.skip :]

        real_signals = rotate2real(signals).real

        if options.dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, pCov) = fit_dual_decay(
                xs, real_signals
            )
            names = ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2")
        else:
            t1, t1err, y_fit, (pOpt, pCov) = fit_decay(xs, real_signals)
            names = ("y0", "yscale", "decay_time")
            t1b = 0.0
            t1berr = 0.0
        quality = compute_fit_quality(
            real_signals,
            y_fit,
            {name: float(value) for name, value in zip(names, pOpt, strict=True)},
            pCov,
        )

        fig, ax = plots.subplots("fit")

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if options.dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            title = f"$T_1$ = {t1_str}, " + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return T1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if options.dual_exp else None,
            t1b_err=float(t1berr) if options.dual_exp else None,
            fit_quality={"fit": quality},
        )
