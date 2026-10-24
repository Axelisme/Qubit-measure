from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import zcu_tools.analysis.fitting as ft
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_dual_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting import live_heatmap_with_line, live_line
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
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", group="Modules")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse", group="Modules")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout", group="Modules")]
    length: Annotated[
        SweepCfg | list[float], FieldMeta(label="Delay", unit="us", group="Sweep")
    ]
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = True


@dataclass(frozen=True)
class T1AnalyzeOptions:
    dual_exp: bool = False
    skip: int = 0


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
                modules = cfg
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
                        Reset("reset", modules.reset),
                        Pulse("pi_pulse", modules.pi_pulse),
                        Delay("t1_delay", length_param),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(rounds=cfg.rounds)
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
                modules = cfg

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
                        Reset("reset", modules.reset),
                        Pulse("pi_pulse", modules.pi_pulse),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length_idx", len(length_cycles))
                    .build_and_acquire(rounds=cfg.rounds)
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(group="Modules")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(group="Modules")]
    test_pulse: Annotated[Pulse.Cfg, FieldMeta(group="Modules")]
    readout: Annotated[Readout.Cfg, FieldMeta(group="Modules")]
    length: Annotated[SweepCfg, FieldMeta(group="Sweep")]


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
        modules = cfg

        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
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
            modules = cfg

            # 2. Sweep the tone after excitation, then acquire the readout IQ.
            length_sweep = cfg.length
            length_param = sweep2param("length", length_sweep)
            modules.test_pulse.set_param("length", length_param)

            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse(name="pi_pulse", cfg=modules.pi_pulse),
                    Pulse(name="test_pulse", cfg=modules.test_pulse),
                    Readout("readout", modules.readout),
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


@dataclass(frozen=True)
class ScanT1WithToneResult:
    values: NDArray[np.float64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_with_tone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ScanT1WithToneCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(group="Modules")] = None
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(group="Modules")]
    test_pulse: Annotated[Pulse.Cfg, FieldMeta(group="Modules")]
    readout: Annotated[Readout.Cfg, FieldMeta(group="Modules")]
    gain: Annotated[SweepCfg | list[float], FieldMeta(group="Sweep")]
    length: Annotated[SweepCfg, FieldMeta(group="Sweep")]


@dataclass(frozen=True)
class ScanT1WithToneAnalysis:
    gains: NDArray[np.float64]
    t1s: NDArray[np.float64]
    t1errs: NDArray[np.float64]


class ScanT1WithToneExp(Experiment[ScanT1WithToneResult, ScanT1WithToneCfg]):
    # inner = times (result_shape, s on disk -> US_TO_S); outer = values (gain, a.u.)
    AXES_SPEC: ClassVar[AxesSpec[ScanT1WithToneResult, ScanT1WithToneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("times", "Time", "s", 1e-6),
            AxesSpec.Axis("values", "Readout Gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=ScanT1WithToneResult,
        cfg_type=ScanT1WithToneCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/t1_with_tone_sweep",
    )

    def run(
        self, cfg: ScanT1WithToneCfg, *, context: RunContext
    ) -> ScanT1WithToneResult:
        """Measure tone-driven relaxation versus test-tone gain and duration.

        cfg supplies reset, π, test tone and readout modules, host-scanned gains
        and a hardware duration sweep in us. context supplies acquisition and
        live plots. At each gain apply reset → π → test tone → readout.
        Return quantized gains, durations and mean complex IQ with shape
        (gain, time), keeping NaN for unmeasured points. Invalid pulse or sweep
        settings propagate validation or compilation errors.
        """
        # 1. Prepare the host gain axis and the generator's tone-duration axis.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
            allow_array=True,
        )
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
        )

        viewer = live_heatmap_with_line(
            context.plots, "measurement", "gain", "Time (us)", line_axis=1, num_lines=5
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=lambda data: viewer.update(
                gains, lengths, t1_with_tone_signal2real(data)
            ),
        )

        # 2. Scan gains on the host; sweep tone duration inside each program.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                cfg = step.cfg
                modules = cfg

                modules.test_pulse.set_param("gain", step.value)
                length_sweep = cfg.length
                length_param = sweep2param("length", length_sweep)
                modules.test_pulse.set_param("length", length_param)

                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("pi_pulse", modules.pi_pulse),
                        Pulse("test_pulse", modules.test_pulse),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        # 3. Preserve gain-major IQ averages and unfinished NaN slots.
        return ScanT1WithToneResult(
            values=gains,
            times=lengths,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[ScanT1WithToneCfg, ScanT1WithToneResult],
        options: None,
        *,
        plots: Plots,
    ) -> ScanT1WithToneAnalysis:
        result = source.result
        del options

        gains, ts, signals = result.values, result.times, result.signals

        signals = np.asarray(gaussian_filter(signals, sigma=1), dtype=np.complex128)
        real_signals = t1_with_tone_signal2real(signals)

        t1s = np.full(len(gains), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(gains)):
            real_signal = real_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signal)):
                continue

            t1, t1err, *_ = fit_decay(ts, real_signal)

            if t1err > 0.3 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        gains = gains[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        fig, _ = plots.subplots("fit", nrows=2, ncols=1, sharex=True)
        ax1, ax2 = fig.axes

        fig.suptitle("T1 while readout")

        ax1.set_ylabel("T1 over sweep value")
        ax1.imshow(
            real_signals.T,
            aspect="auto",
            extent=(gains[0], gains[-1], t1s[0], t1s[-1]),
            origin="lower",
        )
        ax2.errorbar(
            gains, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Readout Gain (a.u.)")
        ax2.set_ylabel("T1 (us)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(gains[0], gains[-1])
        ax2.grid()

        return ScanT1WithToneAnalysis(gains=gains, t1s=t1s, t1errs=t1errs)
