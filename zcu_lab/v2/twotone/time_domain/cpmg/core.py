from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
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


@dataclass(frozen=True)
class CPMG_Result:
    """Mean complex IQ for each pi-pulse count and total evolution time.

    ns lists pi-pulse counts. delays holds each count's quantized time grid in
    us; signals has the same (count, time index) shape. IQ is not population.
    """

    ns: NDArray[np.int64]
    delays: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class CPMGAnalyzeOptions:
    """Fringe fitting and optional reference coherence times in us."""

    fit_fringe: Annotated[bool, FieldMeta(label="Fit fringes")] = True
    t2r: Annotated[float | None, FieldMeta(label="Ramsey T2 reference (µs)")] = None
    t1: Annotated[float | None, FieldMeta(label="T1 reference (µs)")] = None


@dataclass(frozen=True)
class CPMGAnalysis:
    """Weighted T2(n) = T2_1 * n**gamma summary; times and errors are in us.

    fit_rows counts accepted rows with finite positive n, T2 and uncertainty.
    Fewer than three usable rows leaves all four fit values None.
    """

    t2_1_us: float | None
    t2_1_err_us: float | None
    gamma: float | None
    gamma_err: float | None
    fit_rows: int


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project mean IQ and scale each count's observed range, keeping NaN holes."""
    real_signals = rotate2real(signals).real
    max_vals = np.fmax.reduce(real_signals, axis=1, keepdims=True, initial=np.nan)
    min_vals = np.fmin.reduce(real_signals, axis=1, keepdims=True, initial=np.nan)
    return (real_signals - min_vals) / np.clip(max_vals - min_vals, 1e-12, None)


class CPMG_Cfg(ConfigBase):
    reset: Reset.Cfg | None = None
    pi2_pulse: Pulse.Cfg
    pi_pulse: Pulse.Cfg
    readout: Readout.Cfg
    count: Annotated[
        list[Annotated[int, Field(strict=True, ge=1)]],
        Field(min_length=1),
        FieldMeta(label="Pi pulse counts", unit="count"),
    ]
    delay: SweepCfg | list[SweepCfg]  # Each pulse count can use its own delay sweep.
    detune_ratio: Annotated[
        float, FieldMeta(label="Detune ratio (fringes/step)", decimals=3)
    ] = 0.0
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]
    earlystop_snr: Annotated[float | None, FieldMeta(label="Early-stop SNR")] = None


def _summarize_t2(
    ns: NDArray[np.int64],
    t2s: NDArray[np.float64],
    t2errs: NDArray[np.float64],
) -> CPMGAnalysis:
    """Return the weighted log-log T2 summary without changing inputs or plots.

    Pass aligned pulse counts, fitted T2 values and their absolute errors in us.
    Only finite positive rows contribute. Fewer than three usable rows return
    None coefficients and their row count; otherwise return the power-law
    coefficients and errors from the unscaled covariance. NumPy fitting errors
    propagate.
    """
    usable = (
        np.isfinite(ns)
        & (ns > 0)
        & np.isfinite(t2s)
        & (t2s > 0)
        & np.isfinite(t2errs)
        & (t2errs > 0)
    )
    fit_rows = int(np.count_nonzero(usable))
    if fit_rows < 3:
        return CPMGAnalysis(None, None, None, None, fit_rows)
    (gamma, intercept), covariance = np.polyfit(
        np.log(ns[usable]),
        np.log(t2s[usable]),
        1,
        w=t2s[usable] / t2errs[usable],
        cov="unscaled",
    )
    t2_1 = float(np.exp(intercept))
    return CPMGAnalysis(
        t2_1_us=t2_1,
        t2_1_err_us=t2_1 * float(np.sqrt(covariance[1, 1])),
        gamma=float(gamma),
        gamma_err=float(np.sqrt(covariance[0, 0])),
        fit_rows=fit_rows,
    )


CPMG_LENGTHS_VARIABLE = "lengths"
CPMG_SIGNALS_VARIABLE = "signals"
CPMG_GROUPED_VARIABLES = (CPMG_LENGTHS_VARIABLE, CPMG_SIGNALS_VARIABLE)


def cpmg_result_to_grouped_payloads(result: CPMG_Result) -> dict[str, LabberPayload]:
    """Map CPMG count and time arrays to the declared grouped storage payloads."""
    return CPMG_GROUPED_AXES_SPEC.payloads_from_result(result)


def _validate_cpmg_arrays(
    times: NDArray[np.int64],
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> None:
    """Reject saved arrays that do not match this experiment's result layout."""
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
    """Reject saved arrays that do not match this experiment's result layout."""
    _validate_cpmg_arrays(
        np.asarray(result.ns, dtype=np.int64),
        np.asarray(result.delays, dtype=np.float64),
        np.asarray(result.signals, dtype=np.complex128),
    )


def _build_cpmg_result(data: GroupedLoadData[CPMG_Cfg]) -> CPMG_Result:
    """Restore the experiment result from its saved variables and axis coordinates."""
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
    VariableAxisSpec(
        label="Number of Pi", unit="a.u.", field_name="ns", dtype=np.int64
    ),
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
    """Round total-time grids and raise ValueError for too-short pulse intervals.

    Pass the run cfg, pi-pulse counts and board timing configuration. Return the
    per-count SweepCfg windows and a (count, time) array of rounded lengths in
    us. Windows retain cfg references; no input is changed and no plot is made.
    """
    pi2_pulse, pi_pulse = cfg.pi2_pulse, cfg.pi_pulse
    windows = [cfg.delay] * len(times) if isinstance(cfg.delay, SweepCfg) else cfg.delay
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


def _delay_edges(delays: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bound each delay cell by midpoints on its own sweep, in us."""
    if delays.shape[1] == 1:
        return delays + np.array([-0.5, 0.5])
    half_steps = 0.5 * np.diff(delays, axis=1)
    return np.concatenate(
        (
            delays[:, :1] - half_steps[:, :1],
            delays[:, :-1] + half_steps,
            delays[:, -1:] + half_steps[:, -1:],
        ),
        axis=1,
    )


def _count_map(
    ax: Axes,
    counts: NDArray[np.int64],
    delays: NDArray[np.float64],
    contrast: NDArray[np.float64],
) -> PolyCollection:
    """Draw equally wide count slots; each slot uses its own delay cells and colors."""
    slots = np.arange(counts.size)
    edges = _delay_edges(delays)
    left = np.broadcast_to(slots[:, None] - 0.5, delays.shape)
    right = left + 1
    start, end = edges[:, :-1], edges[:, 1:]
    corners = np.stack(
        (
            np.stack((left, start), axis=-1),
            np.stack((right, start), axis=-1),
            np.stack((right, end), axis=-1),
            np.stack((left, end), axis=-1),
        ),
        axis=-2,
    )
    # Adjacent columns can have different y boundaries; no cell spans their join.
    image = PolyCollection(
        list(corners.reshape(-1, 4, 2)),
        array=contrast.ravel(),
        norm=Normalize(0, 1),
        cmap="RdBu_r",
        edgecolors="none",
        antialiaseds=False,
    )
    ax.add_collection(image, autolim=False)
    duplicates = Counter(counts.tolist())
    ticks = np.linspace(0, counts.size - 1, min(counts.size, 6), dtype=int)
    labels = [
        f"{counts[i]}\n(#{i + 1})" if duplicates[counts[i]] > 1 else str(counts[i])
        for i in ticks
    ]
    ax.set_xticks(ticks, labels=labels)
    ax.set(xlim=(-0.5, counts.size - 0.5), ylim=(edges.min(), edges.max()))
    ax.grid(False)
    return image


def _live_plot(
    plots: Plots, times: NDArray[np.int64], delays: NDArray[np.float64]
) -> Callable[[tuple[NDArray[np.float64], float]], None]:
    """Show each count's delay grid; return an owner-thread contrast/SNR updater.

    The callback receives normalized (count, time) contrast and current SNR.
    It updates the map, two recent traces and titles without changing inputs.
    """
    fig, _ = plots.subplots(1, 2, num="measurement", layout="constrained")
    heat_ax, line_ax = fig.axes
    heat_ax.set(
        title="CPMG acquisition",
        xlabel="π pulses (count)",
        ylabel="Total evolution delay (µs)",
    )
    line_ax.set(
        xlabel="Total evolution delay (µs)", ylabel="Normalized IQ projection (a.u.)"
    )
    heatmap = _count_map(heat_ax, times, delays, np.full_like(delays, np.nan))
    fig.colorbar(heatmap, ax=heat_ax, label="Normalized IQ (a.u.)")
    line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    (previous,) = line_ax.plot(
        [], [], "-", alpha=0.6, color="C1", label="Previous count"
    )
    (latest,) = line_ax.plot(
        [], [], ".-", markersize=5, color="C0", label="Latest count"
    )
    line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))

    @plots.on_plot_thread
    def update(data: tuple[NDArray[np.float64], float]):
        """Refresh the map, recent traces and SNR title; return changed artists."""
        real, snr = data
        valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
        current = int(valid[-1]) if valid.size else -1
        heatmap.set_array(real.ravel())
        for line, slot in ((previous, current - 1), (latest, current)):
            if slot >= 0:
                line.set_data(delays[slot], real[slot])
            else:
                line.set_data([], [])
        line_ax.relim()
        line_ax.autoscale_view()

        title = f"CPMG · {valid.size}/{times.size} counts measured"
        row = f"Latest: {times[current]} π pulses" if current >= 0 else "Awaiting data"
        if snr:
            row += f" · SNR {snr:.1f}"
        return (
            heatmap,
            previous,
            latest,
            heat_ax.set_title(title),
            line_ax.set_title(row),
        )

    fig.show()
    return update


def _plot_coherence(
    plots: Plots,
    result: CPMG_Result,
    lifetimes: NDArray[np.float64],
    errors: NDArray[np.float64],
    summary: CPMGAnalysis,
    options: CPMGAnalyzeOptions,
) -> None:
    """Compare the IQ map with the accepted row fits and optional T1/T2* references."""
    counts, delays = result.ns, result.delays
    contrast = cpmg_signal2real(result.signals)
    fig, _ = plots.subplots(num="fit", nrows=1, ncols=2, figsize=(9, 4))
    ax1, ax2 = fig.axes

    image = _count_map(ax1, counts, delays, contrast)
    fig.colorbar(image, ax=ax1, label="Normalized IQ (a.u.)")
    ax1.set(
        title="CPMG contrast",
        xlabel="π pulses (count)",
        ylabel="Total evolution delay (µs)",
    )

    ax2.set_title(f"CPMG coherence · {summary.fit_rows}/{counts.size} fits")
    ax2.errorbar(
        counts, lifetimes, yerr=errors, label="T₂ ± fit error", color="red", fmt="."
    )
    if summary.t2_1_us is not None and summary.gamma is not None:
        fit_ns = np.geomspace(counts.min(), counts.max(), 200)
        ax2.plot(
            fit_ns,
            summary.t2_1_us * fit_ns**summary.gamma,
            label=r"$T_{2,1} n^{\gamma}$",
            color="black",
        )
        ax2.set_title(f"CPMG coherence · γ = {summary.gamma:.3g}")
    if options.t1 is not None:
        ax2.axhline(2 * options.t1, color="blue", linestyle="--", label=r"$2T_1$")
    if options.t2r is not None:
        ax2.axhline(options.t2r, color="green", linestyle="--", label=r"$T_{2r}$")
    ax2.set_xscale("log")
    if counts.size <= 5:
        ax2.set_xticks(counts, labels=[str(n) for n in counts])
    ax2.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.set_xlabel("π pulses (count)")
    ax2.set_ylabel("CPMG coherence time T₂ (µs)")

    fig.tight_layout()


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
            """Record the latest checked SNR for the live plot title."""
            nonlocal current_snr
            current_snr = snr

        update = _live_plot(context.plots, times, lengths)
        signals_buffer = SignalBuffer(
            (len(times), len(length_idxs)),
            on_update=lambda data: update_viewer(data),
        )
        # 3. Measure refocused coherence versus evolution time for each pi-pulse count.
        with Schedule(signals_buffer, scope=context.scope) as sched:

            def update_viewer(data: NDArray[np.complex128]) -> None:
                """Refresh contrast and recent traces on their own time grids."""
                update((cpmg_signal2real(data), current_snr))

            # Scan pi-pulse counts on the host; sweep evolution time on the board.
            for idx, (time_value, step) in enumerate(sched.scan("count", times)):
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
        Return the weighted log-linear power law T2(n)=T2_1*n**gamma and its
        unscaled propagated errors. Only accepted rows with finite positive
        n, T2 and uncertainty are usable. Fewer than three usable rows gives
        None fit values without failing or changing the plotted T2 points.
        """
        # 1. Separate real-IQ decay fits from normalized contrast used to compare pulse counts.
        result = source.result
        fit_fringe = options.fit_fringe

        times, lengths, signals2D = result.ns, result.delays, result.signals

        real_signals2D = rotate2real(signals2D).real

        def is_good_fit(
            *,
            lens: NDArray[np.float64],
            fit_signal: NDArray[np.float64],
            real_signal: NDArray[np.float64],
            t2r: float,
            t2err: float,
        ) -> bool | np.bool_:
            """Return whether decay time, uncertainty and residual meet fit limits.

            Pass zero-origin times in us, fitted/measured real IQ and the fitted
            decay time/error in us. Return a boolean without mutation or plotting.
            """
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            mean_residual = np.mean(np.abs(smooth_residual))
            return (
                t2r < np.max(lens) and t2err < 0.5 * t2r and fit_ptp > 4 * mean_residual
            )

        # 2. Fit each complete curve relative to its own first sampled time.
        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        for i, (real_signal, lens) in enumerate(
            zip(real_signals2D, lengths, strict=False)
        ):
            if np.any(np.isnan(real_signal)):
                continue

            plens = lens - lens[0]
            fit = (
                fit_decay_fringe(plens, real_signal)
                if fit_fringe
                else fit_decay(plens, real_signal)
            )
            t2 = fit.decay_time
            t2err = fit.decay_time_err
            fit_signal = fit.curve

            if is_good_fit(
                lens=plens,
                fit_signal=fit_signal,
                real_signal=real_signal,
                t2r=t2,
                t2err=t2err,
            ):
                t2s[i] = t2
                t2errs[i] = t2err

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        summary = _summarize_t2(times, t2s, t2errs)

        _plot_coherence(plots, result, t2s, t2errs, summary, options)
        return summary
