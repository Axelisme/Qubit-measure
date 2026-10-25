"""T1 relaxation with a fixed-gain background tone."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import zcu_tools.analysis.fitting as ft
from zcu_tools.analysis.fitting import (
    fit_decay,
    fit_dual_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    sweep2param,
)

from zcu_lab.v2.support import live_line

from .core import T1Analysis, T1Result, t1_signal2real


class T1WithToneCfg(ConfigBase):
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
    pi_pulse: Pulse.Cfg
    test_pulse: Pulse.Cfg
    readout: Readout.Cfg
    length: SweepCfg


@dataclass(frozen=True)
class T1WithToneAnalyzeOptions:
    dual_exp: bool = False


class T1WithToneExp(Experiment[T1Result, T1WithToneCfg]):
    Options: ClassVar[type[T1WithToneAnalyzeOptions]] = T1WithToneAnalyzeOptions

    # times stored as seconds on disk -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1WithToneCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", "Time", "s", 1e-6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1WithToneCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/t1_with_tone",
    )

    def run(self, cfg: T1WithToneCfg, *, context: RunContext) -> T1Result:
        """Measure relaxation while applying a test tone after the π pulse.

        cfg supplies reset, π, test tone and readout modules plus a tone-length
        sweep in us. context supplies acquisition and live plots. Sweep the
        tone duration on its generator grid, then read out; return quantized
        durations and mean complex IQ with shape (time,), keeping NaN for
        unmeasured points. Invalid pulse or sweep settings propagate validation
        or compilation errors.
        """
        # 1. Prepare tone durations on the test generator's grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.test_pulse.ch},
        )

        viewer = live_line(
            context.plots,
            "measurement",
            "Time (us)",
            "Amplitude",
            title="T1 relaxation",
        )
        signals_buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, t1_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            cfg = sched.cfg

            # 2. Sweep the tone after excitation, then acquire the readout IQ.
            length_sweep = cfg.length
            length_param = sweep2param("length", length_sweep)
            cfg.test_pulse.set_param("length", length_param)

            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", cfg.reset),
                    Pulse(name="pi_pulse", cfg=cfg.pi_pulse),
                    Pulse(name="test_pulse", cfg=cfg.test_pulse),
                    Readout("readout", cfg.readout),
                )
                .declare_sweep("length", length_sweep)
                .build_and_acquire(rounds=cfg.rounds)
            )

        # 3. Return acquired averages without filling unfinished points.
        return T1Result(times=lengths, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[T1WithToneCfg, T1Result],
        options: T1WithToneAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1Analysis:
        result = source.result
        dual_exp = options.dual_exp

        xs, signals = result.times, result.signals

        real_signals = t1_signal2real(signals)

        if dual_exp:
            t1b, t1berr, t1, t1err, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)
            t1b = 0.0
            t1berr = 0.0

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us" if dual_exp else "N/A"

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(xs, y_fit, label="fit")
        if dual_exp:
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
        else:
            ax.set_title(f"T1 = {t1_str}", fontsize=15)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return T1Analysis(
            t1=float(t1),
            t1_err=float(t1err),
            t1b=float(t1b) if dual_exp else None,
            t1b_err=float(t1berr) if dual_exp else None,
        )
