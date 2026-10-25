"""Measure T1 from this point's pulses, settings and lifetime guess."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.support import LinePlot, live_line
from zcu_lab.v2.support.lifetime_models import (
    LifetimeComparison,
    LifetimeModelRecord,
    draw_lifetime_comparison,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_lab.workflows.coherence_fluxdep.pulses import CalibratedPulses
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Tunables
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    analyze_t1,
    compare_decay_fit,
    lifetime_rejection,
)
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gates import Gate, gate
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved


@dataclass(frozen=True, kw_only=True)
class T1Result:
    """T1 outcome, independent of the dephasing experiments' result types.

    kind is always t1. status is accepted, rejected, failed or skipped. reason
    is None on acceptance, otherwise the dependency, acquisition or fit reason.
    lifetime_us/error_us contain accepted microseconds only, otherwise None.
    file is t1.h5 after saving a Completed, including partial/rejected traces;
    acquisition failure and dependency skips have no file. detune_mhz remains
    None for T1 and preserves the existing record shape.
    models records both diagnostic fits, or is empty before primary-fit success.
    gates records named comparisons after primary-fit success, otherwise empty.
    Rejection reasons start with lifetime-quality diagnostics, then list failed
    gates with actual values, failed comparisons and thresholds.
    Partial saves raw then returns rejected/stopped without fitting. Expected
    fit failure returns failed with the saved file; unexpected errors propagate.
    Construct with keywords; this result never changes point records or history.
    """

    kind: Literal["t1"] = "t1"
    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    detune_mhz: float | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()


def _cfg(
    pulses: CalibratedPulses, settings: T1Tunables, *, points: int, guess_us: float
) -> T1Cfg:
    if not math.isfinite(guess_us) or guess_us <= 0:
        raise ValueError("T1 lifetime guess must be positive and finite")
    stop = (
        min(
            settings.max_length_us,
            max(settings.sweep_stop_min_us, settings.sweep_factor * guess_us),
        )
        if settings.sweep_mode == "auto"
        else settings.sweep_stop_us
    )
    relax = (
        max(settings.relax_min_us, settings.relax_factor * guess_us)
        if settings.relax_mode == "auto"
        else settings.relax_delay_us
    )
    return T1Cfg(
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        earlystop_snr=settings.earlystop_snr,
        relax_delay=relax,
        uniform=settings.uniform,
        pi_pulse=pulses.pi.model_copy(deep=True),
        readout=pulses.readout.model_copy(deep=True),
        length=SweepCfg(
            start=settings.sweep_start_us,
            stop=stop,
            expts=points,
            step=(stop - settings.sweep_start_us) / (points - 1),
        ),
    )


def _plot(plots: Plots) -> LinePlot:
    if "t1" not in plots:
        plots.subplots("t1", nrows=2)

    def rebuild(figure: Figure) -> None:
        figure.set_layout_engine("constrained")
        # Give this point its own handle, not the previous live template's axis.
        figure.axes[1].remove()
        figure.add_subplot(2, 1, 2)

    plots.draw("t1", rebuild)
    return live_line(
        plots, "t1", "Time (us)", "Signal (raw I)", axes=plots["t1"].axes[1]
    )


def _draw_fit(
    plots: Plots, fit: DecayFit, comparison: LifetimeComparison, reason: str | None
) -> None:
    def overlay(figure: Figure) -> None:
        ax = figure.axes[1]
        ax.clear()
        ax.plot(fit.times, fit.observed, ".", label="observed")
        draw_lifetime_comparison(ax, fit.times, comparison)
        ax.set(
            xlabel="Time (us)",
            ylabel="Signal (projected)",
            title=(
                f"t1 = {fit.lifetime_us:.2f} ± {fit.error_us:.2f} us"
                + ("\nRejected fit" if reason is not None else "")
            ),
        )
        ax.legend()

    plots.draw("t1", overlay)


async def measure_t1(
    scope: Scope,
    env: StageEnv,
    pulses: CalibratedPulses,
    settings: T1Tunables,
    *,
    points: int,
    guess_us: float,
) -> T1Result:
    """Acquire energy relaxation, save raw, fit, gate and draw both T1 models.

    pulses supplies this point's accepted pi and retained readout. settings
    controls averaging, timing and acceptance. points is the delay sample count
    (at least five); guess_us is the positive finite history estimate in us.
    Auto timing clamps sweep_factor*guess_us and floors relax_factor*guess_us;
    fixed timing uses the explicit settings unchanged. Inputs remain unchanged.
    env supplies workflow plots, captured save identity and summary observation.
    Save t1.h5 before fitting; partial returns rejected/stopped without fitting.
    Acquisition failure returns failed without a file; expected fit failure
    returns failed with the saved file. Other exceptions propagate. Only the
    primary fit controls acceptance. Rejection reasons retain lifetime_rejection
    first, followed by each failed numeric gate. No point record or history is modified.
    """
    cfg = _cfg(pulses, settings, points=points, guess_us=guess_us)
    plot = _plot(env.plots)
    acquisition = await acquire_saved(
        scope,
        env,
        T1Exp().run,
        cfg,
        name="T1",
        output=NativeOutput(filename="t1.h5", spec=T1Exp.AXES_SPEC),
    )
    if acquisition.result is None:
        return T1Result(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            file=acquisition.file,
        )
    plot.update(acquisition.result.times, acquisition.result.signals.real)
    fit = analyze_t1(acquisition.result)
    if isinstance(fit, Failed):
        return T1Result(status="failed", reason=fit.reason, file=acquisition.file)

    gates = (
        gate("positive_lifetime_us", ">", 0, fit.lifetime_us),
        gate("nonnegative_error_us", ">=", 0, fit.error_us),
        gate(
            "relative_error",
            "<",
            settings.max_rel_err,
            fit.error_us / fit.lifetime_us if fit.lifetime_us > 0 else None,
        ),
        gate(
            "normalized_residual_rms",
            "<",
            settings.max_fit_residual,
            fit.quality.normalized_residual_rms,
        ),
        gate(
            "lifetime_window_us",
            "<=",
            2.0 * float(np.max(acquisition.result.times)),
            fit.lifetime_us,
        ),
        gate(
            "mean_absolute_fit_error",
            "<=",
            0.1 * float(np.ptp(fit.fitted)),
            float(np.mean(np.abs(fit.observed - fit.fitted))),
        ),
        gate("invalid_quality_count", "==", 0, len(fit.quality.invalid)),
        gate(
            "points_within_lifetime",
            ">=",
            settings.min_lifetime_points,
            int(np.count_nonzero((fit.times >= 0) & (fit.times <= fit.lifetime_us))),
        ),
    )
    reasons = []
    lifetime_reason = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, settings.min_lifetime_points
    )
    if lifetime_reason is not None:
        reasons.append(lifetime_reason)
    failed_comparison = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!="}
    reasons.extend(
        f"{item.name} {item.actual:.9g} {failed_comparison[item.comparison]} {item.threshold:.9g}"
        if item.actual is not None
        else f"{item.name} unavailable (requires {item.comparison} {item.threshold:.9g})"
        for item in gates
        if not item.passed
    )
    reason = "; ".join(reasons) or None
    comparison = compare_decay_fit(
        fit, t1=True, max_window_factor=settings.max_model_window_factor
    )
    _draw_fit(env.plots, fit, comparison, reason)
    if reason is not None:
        return T1Result(
            status="rejected",
            reason=reason,
            file=acquisition.file,
            models=comparison.records,
            gates=gates,
        )
    return T1Result(
        status="accepted",
        lifetime_us=fit.lifetime_us,
        error_us=fit.error_us,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
    )
