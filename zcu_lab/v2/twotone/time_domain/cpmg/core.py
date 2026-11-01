from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as validated_dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from pydantic import Field
from qick import QickConfig
from scipy.ndimage import gaussian_filter1d
from zcu_tools.analysis.fitting import fit_decay, fit_decay_fringe
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import LabberPayload
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import make_sweep
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap, live_line


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / np.clip(max_vals - min_vals, 1e-12, None)


@dataclass(frozen=True)
class CPMG_Result:
    """Mean complex IQ for each pi-pulse count and total evolution time.

    ns lists pi-pulse counts. delays holds each count's quantized time grid in
    us; signals has the same (count, time index) shape. IQ is not population.
    """

    ns: NDArray[np.int64]
    delays: NDArray[np.float64]
    signals: NDArray[np.complex128]


class CPMG_Cfg(ConfigBase):
    reset: Reset.Cfg | None = None
    pi2_pulse: Pulse.Cfg
    pi_pulse: Pulse.Cfg
    readout: Readout.Cfg
    count: Annotated[
        list[Annotated[int, Field(strict=True, ge=1)]],
        Field(min_length=1),
        FieldMeta(label="Pi pulse counts"),
    ]
    delay: SweepCfg | list[SweepCfg]
    detune_ratio: Annotated[
        float, FieldMeta(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.0
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
    earlystop_snr: Annotated[float | None, FieldMeta(label="Early-stop SNR")] = None


@validated_dataclass(frozen=True)
class CPMGAnalyzeOptions:
    """Fringe fitting and optional reference coherence times in us."""

    fit_fringe: Annotated[bool, FieldMeta(label="Fit fringes")] = True
    t2r: Annotated[float | None, FieldMeta(label="Ramsey T2 reference (us)")] = None
    t1: Annotated[float | None, FieldMeta(label="T1 reference (us)")] = None


@dataclass(frozen=True)
class CPMGAnalysis:
    """T2 values and errors in microseconds, indexed by pulse count."""

    ns: NDArray[np.int64]
    t2s: NDArray[np.float64]
    t2errs: NDArray[np.float64]


CPMG_LENGTHS_VARIABLE = "lengths"
CPMG_SIGNALS_VARIABLE = "signals"
CPMG_GROUPED_VARIABLES = (CPMG_LENGTHS_VARIABLE, CPMG_SIGNALS_VARIABLE)


def cpmg_result_to_grouped_payloads(result: CPMG_Result) -> dict[str, LabberPayload]:
    return CPMG_GROUPED_AXES_SPEC.payloads_from_result(result)


def _validate_cpmg_arrays(
    times: NDArray[np.int64],
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> None:
    if times.ndim != 1:
        raise ValueError(f"CPMG ns must be 1-D, got shape {times.shape}")
    if lengths.ndim != 2:
        raise ValueError(f"CPMG delays must be 2-D, got shape {lengths.shape}")
    if signals.shape != lengths.shape:
        raise ValueError(
            "CPMG signals shape must match delays shape "
            f"(got signals={signals.shape}, delays={lengths.shape})"
        )
    if lengths.shape[0] != times.shape[0]:
        raise ValueError(
            "CPMG delays first dimension must match ns length "
            f"(got delays={lengths.shape}, ns={times.shape})"
        )


def _validate_cpmg_result(result: CPMG_Result) -> None:
    _validate_cpmg_arrays(
        np.asarray(result.ns, dtype=np.int64),
        np.asarray(result.delays, dtype=np.float64),
        np.asarray(result.signals, dtype=np.complex128),
    )


def _build_cpmg_result(data: GroupedLoadData[CPMG_Cfg]) -> CPMG_Result:
    lengths_variable = data.variable(CPMG_LENGTHS_VARIABLE)
    signals_variable = data.variable(CPMG_SIGNALS_VARIABLE)
    ns = lengths_variable.axes[1].astype(np.int64)
    delays = lengths_variable.z.astype(np.float64)
    signals = signals_variable.z.astype(np.complex128)
    _validate_cpmg_arrays(ns, delays, signals)
    return CPMG_Result(
        ns=ns,
        delays=delays,
        signals=signals,
    )


_CPMG_VARIABLE_AXES = (
    VariableAxisSpec.generated_arange(label="Time Index", unit="a.u.", dtype=np.int64),
    VariableAxisSpec(label="Number of Pi", unit="a.u.", field_name="ns", dtype=np.int64),
)
CPMG_GROUPED_AXES_SPEC = GroupedAxesSpec(
    variables=(
        VariableSpec(
            variable=CPMG_LENGTHS_VARIABLE,
            axes=_CPMG_VARIABLE_AXES,
            z=VariableZSpec(
                field_name="delays",
                label="Length",
                unit="s",
                scale=1e-6,
                dtype=np.float64,
            ),
        ),
        VariableSpec(
            variable=CPMG_SIGNALS_VARIABLE,
            axes=_CPMG_VARIABLE_AXES,
            z=VariableZSpec(
                field_name="signals",
                label="Signal",
                unit="a.u.",
                dtype=np.complex128,
            ),
        ),
    ),
    result_type=CPMG_Result,
    cfg_type=CPMG_Cfg,
    cfg_schema_version="4.0",
    tag="twotone/ge/cpmg",
    result_builder=_build_cpmg_result,
    result_validator=_validate_cpmg_result,
)


def _prepare_delays(
    cfg: CPMG_Cfg, times: NDArray[np.int64], soccfg: QickConfig
) -> tuple[list[SweepCfg], NDArray[np.float64]]:
    """Round each count's total-time grid and reject intervals too short for pulses."""
    pi2_pulse, pi_pulse = cfg.pi2_pulse, cfg.pi_pulse
    windows = [cfg.delay] * len(times) if isinstance(cfg.delay, SweepCfg) else cfg.delay
    if len(windows) != len(times):
        raise ValueError("Provide one delay window or one per pi-pulse count")
    if len({window.expts for window in windows}) != 1:
        raise ValueError("All delay windows must have the same number of points")
    lengths = np.array(
        [
            sweep2array(window, "time", {"soccfg": soccfg, "scaler": 1 / (2 * count)})
            for window, count in zip(windows, times, strict=True)
        ]
    )

    # Reject timing that cannot fit the preparation and refocusing pulses.
    min_interval = min(
        window.start / (2 * count) for window, count in zip(windows, times, strict=True)
    )
    if min_interval < 0.5 * (  # the first and last delays are half of the interval
        pi2_pulse.waveform.length + pi_pulse.waveform.length
    ):
        raise ValueError(
            "The interval is too short to measure the CPMG signal",
            f"min_interval: {min_interval:.2g}, ",
            f"pi2_pulse_length: {pi2_pulse.waveform.length:.2g}, ",
            f"pi_pulse_length: {pi_pulse.waveform.length:.2g}",
        )

    return windows, lengths


def _live_plot(
    plots: Plots, times: NDArray[np.int64]
) -> Callable[[tuple[NDArray[np.float64], NDArray[np.float64], float]], None]:
    """Show the normalized count/time map and its two most recent scan rows."""
    fig, _ = plots.subplots(1, 2, num='measurement')
    heat_ax, line_ax = fig.axes
    heat_ax.set(xlabel='Number of Pi', ylabel='Time idxs')
    line_ax.set_xlabel('Time idxs')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    heat_ax.xaxis.set_major_formatter(formatter)
    heat_ax.yaxis.set_major_formatter(formatter)
    heatmap = live_heatmap(heat_ax, uniform=True)
    line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    line_ax.grid()
    lines = live_line(line_ax, num_lines=2)
    for artist in line_ax.lines[:-1]:
        artist.set(marker="None", alpha=0.3, color="red")
    line_ax.lines[-1].set(label="current line", marker=".", color="C0")
    line_ax.legend()

    @plots.on_plot_thread
    def update(data: tuple[NDArray[np.float64], NDArray[np.float64], float]):
        ys, real, snr = data
        valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
        current = int(valid[-1]) if valid.size else -1
        history = np.full((2, len(ys)), np.nan)
        for offset in range(min(2, current + 1)):
            history[-offset - 1] = real[current - offset]
        artists = heatmap.set(times.astype(np.float64), ys, real)
        artists = (*artists, *lines.set(ys, history))
        title = f'snr = {snr:.1f}' if snr else None
        if title is not None:
            return (*artists, heat_ax.set_title(title))
        return artists

    fig.show()
    return update


class CPMG_Exp(Experiment[CPMG_Result, CPMG_Cfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CPMG_Result, CPMG_Cfg]] = CPMG_GROUPED_AXES_SPEC

    Options: ClassVar[type[CPMGAnalyzeOptions]] = CPMGAnalyzeOptions

    def run(
        self,
        cfg: CPMG_Cfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> CPMG_Result:
        """Measure coherence decay versus refocusing-pulse count and time.

        After optional reset, apply pi/2, a train of n pi pulses, then a final
        pi/2 projection and readout. The first and last timing intervals include
        the finite pi and pi/2 pulse-width correction. detune_ratio advances the
        final projection phase by 360 * detune_ratio degrees per time index.

        cfg.count supplies positive integer pi-pulse counts. delay supplies
        a shared SweepCfg time window in us or one window per count, with equal
        point counts. Too-short intervals or invalid counts raise
        ValueError. acquire_kwargs forwards optional acquisition settings.
        Return each count's quantized times and mean complex IQ; analyze fits
        their decay to estimate T_CPMG. earlystop_snr can stop averaging a row.
        """
        # 1. Prepare pulse counts and each count's hardware-rounded time grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune_ratio, earlystop_snr = cfg.detune_ratio, cfg.earlystop_snr

        pi2_pulse = cfg.pi2_pulse
        pi_pulse = cfg.pi_pulse

        times = np.array(cfg.count, dtype=np.int64)

        windows, lengths = _prepare_delays(cfg, times, soccfg)
        length_idxs = np.arange(windows[0].expts, dtype=np.int64)

        # 2. Track mean IQ and the SNR used to stop averaging each decay curve.
        current_snr = 0.0

        def update_snr(snr: float) -> None:
            nonlocal current_snr
            current_snr = snr

        update = _live_plot(context.plots, times)
        current_length_idx = 0
        signals_buffer = SignalBuffer(
            (len(times), len(length_idxs)),
            on_update=lambda data: update_viewer(data),
        )
        with Schedule(signals_buffer, scope=context.scope) as sched:

            def update_viewer(data: NDArray[np.complex128]) -> None:
                update((lengths[current_length_idx], cpmg_signal2real(data), current_snr))

            # Scan pi-pulse counts on the host; sweep evolution time on the board.
            for idx, (time_value, step) in enumerate(sched.scan("count", times)):
                current_length_idx = idx

                time = int(time_value)
                pi2_pulse = cfg.pi2_pulse
                pi_pulse = cfg.pi_pulse
                dpulse_len = pi_pulse.waveform.length - pi2_pulse.waveform.length

                delay_sweep = windows[idx]
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                delay = builder.add_sweep("delay", delay_sweep)
                detune_param = (
                    360
                    * detune_ratio
                    * sweep2param(
                        "delay", make_sweep(start=0, step=1, expts=delay_sweep.expts)
                    )
                )
                interval = delay / (2 * time)

                # Pulse starts do not consume their duration from the spacing timeline.
                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.fork("default", into="preparation")
                builder.emit(Pulse("pi2_pulse", pi2_pulse), timeline="preparation")
                builder.timeline_inc(interval - 0.5 * dpulse_len)
                builder.delay_by_timeline()
                with builder.repeat("pi_loop", time - 1):
                    builder.fork("default", into="refocus")
                    builder.emit(Pulse("pi_pulse", pi_pulse), timeline="refocus")
                    builder.timeline_inc(2 * interval)
                    builder.delay_by_timeline()
                builder.fork("default", into="last_refocus")
                builder.emit(Pulse("last_pi_pulse", pi_pulse), timeline="last_refocus")
                builder.timeline_inc(interval + 0.5 * dpulse_len)
                builder.delay_by_timeline()
                builder.emit(
                    Pulse(
                        "pi2_projection",
                        pi2_pulse,
                        phase=pi2_pulse.phase + detune_param,
                    )
                )
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(
                    prog,
                    soc,
                    rounds=cfg.rounds,
                    options=RunOptions(
                        stop_condition=snr_checker(
                            signals_buffer[step],
                            earlystop_snr,
                            lambda x: rotate2real(x).real,
                            after_check=update_snr,
                        )
                    ),
                    **(acquire_kwargs or {}),
                )

        # 4. Keep the per-count time grids alongside the unnormalized mean IQ.
        return CPMG_Result(
            ns=times,
            delays=lengths,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[CPMG_Cfg, CPMG_Result],
        options: CPMGAnalyzeOptions,
        *,
        plots: Plots,
    ) -> CPMGAnalysis:
        """Fit T_CPMG in us for each pulse count and plot its count dependence.

        Fit a decay with or without fringes according to fit_fringe. Incomplete
        rows and fits failing the decay-time, uncertainty or contrast checks
        retain NaN T2 values. Raise ValueError if no row passes. Optional t2r
        and t1 values in us add Ramsey and 2*T1 reference lines, not fit inputs.
        Return counts, accepted decay times and their fit uncertainties.
        """
        result = source.result
        fit_fringe, t2r, t1 = options.fit_fringe, options.t2r, options.t1

        times, lengths, signals2D = result.ns, result.delays, result.signals

        real_signals2D = rotate2real(signals2D).real
        norm_signals = cpmg_signal2real(signals2D)

        def is_good_fit(lens, fit_signal, real_signal, t2r, t2err) -> bool:
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            mean_residual = np.mean(np.abs(smooth_residual))
            return (
                t2r < np.max(lens) and t2err < 0.5 * t2r and fit_ptp > 4 * mean_residual
            )

        # Fit each complete curve relative to its own first sampled time.
        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        for i, (real_signal, lens) in enumerate(
            zip(real_signals2D, lengths, strict=False)
        ):
            if np.any(np.isnan(real_signal)):
                continue

            plens = lens - lens[0]
            if fit_fringe:
                fit = fit_decay_fringe(plens, real_signal)
                t2 = fit.decay_time
                t2err = fit.decay_time_err
                fit_signal = fit.curve
            else:
                fit = fit_decay(plens, real_signal)
                t2 = fit.decay_time
                t2err = fit.decay_time_err
                fit_signal = fit.curve

            if is_good_fit(plens, fit_signal, real_signal, t2, t2err):
                t2s[i] = t2
                t2errs[i] = t2err

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        # Compare decay times across pulse counts without treating IQ as population.
        fig, _ = plots.subplots(num="fit", nrows=1, ncols=2, figsize=(8, 3))
        ax1, ax2 = fig.axes

        X = np.broadcast_to(times[None, :], norm_signals.T.shape)
        Y = lengths.T
        ax1.pcolormesh(X, Y, norm_signals.T, shading="nearest")
        ax1.set_xlabel("Number of Pi")
        ax1.set_ylabel("Time (us)")

        ax2.errorbar(times, t2s, yerr=t2errs, label=r"$T_{CPMG}$", color="red", fmt=".")
        if t1 is not None:
            ax2.axhline(2 * t1, color="blue", linestyle="--", label=r"$2T_1$")
        if t2r is not None:
            ax2.axhline(t2r, color="green", linestyle="--", label=r"$T_{2r}$")
        ax2.set_xlim(0.3, None)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.set_xlabel("Number of Pi")

        fig.tight_layout()
        return CPMGAnalysis(ns=times, t2s=t2s, t2errs=t2errs)
