"""Single-fringe fitting policy and plots for idle conditional Ramsey."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fit_decay_fringe
from zcu_tools.plotting.plots import Plots
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class IdleBranchFit:
    """One nominal spectator branch at one DC coordinate.

    dc_index/branch select the original axes (branch 0=I, 1=Xπ). acquired_shots
    counts completed delay/round shots; complete requires all expected samples.
    trace is the T-point mean target IQ rotated to its principal real axis,
    retaining NaN where no round exists. fit_curve is the existing decay-cosine
    model or None when unavailable. magnitude_mhz is nonnegative; signed_mhz
    assumes both branches stay on the side selected by the operator's detune.
    error_mhz is the fitted frequency's covariance 1σ error. resolution_mhz is
    max(1/actual delay span in µs, error_mhz). reason explains why signed_mhz is
    None. fit_quality reports diagnostics even for a sign-ambiguous fitted curve.
    No automatic lack-of-fit/statistical threshold rejects finite fits. Operators
    judge single-frequency model applicability from residuals and uncertainty;
    mixed conditional frequencies can pass. None never means a measured zero.
    Thermal branches are not pure states.
    """

    dc_index: int
    branch: int
    acquired_shots: int
    complete: bool
    trace: NDArray[np.float64]
    fit_curve: NDArray[np.float64] | None = None
    magnitude_mhz: float | None = None
    signed_mhz: float | None = None
    error_mhz: float | None = None
    resolution_mhz: float | None = None
    reason: str | None = None
    fit_quality: FitQuality | None = None


@dataclass(frozen=True)
class IdleZZAnalysis:
    """Ordered (DC-major, I then Xπ) fits and one conditional difference per DC.

    zeta_mhz is f_Xπ−f_I using detune-signed fringe frequencies, never a bare
    magnitude difference. zeta_error_mhz propagates independent branch covariance
    errors in quadrature. Missing/unidentifiable branches leave both None and a
    zeta_reasons explanation. This is a nominal-branch single-model estimate,
    not a purified Hamiltonian ZZ or a writeback/calibration proposal.
    """

    branches: tuple[IdleBranchFit, ...]
    zeta_mhz: tuple[float | None, ...]
    zeta_error_mhz: tuple[float | None, ...]
    zeta_reasons: tuple[str | None, ...]


def fit_branch(
    *,
    dc_index: int,
    branch: int,
    delays: NDArray[np.float64],
    shots: NDArray[np.complex128],
    acquired: NDArray[np.bool_],
    detune: float,
) -> IdleBranchFit:
    """Fit complete (T,rounds,reps) target IQ, preserving incomplete trace points.

    acquired is (T,rounds), established by the core's all-tone finite-round
    validation. Only complete branches enter the existing decay cosine fitter,
    with free phase. At least six distinct, regularly spaced delays are needed
    for five fitted parameters plus residual degrees of freedom. No radius cut,
    heralding, population correction or simulator input is used.
    Sign is detune's sign. In magnitude coordinates the ambiguous boundaries
    are 0 and 2*abs(detune), including for negative detune. A distance no larger
    than max(1/actual sampled delay span, covariance frequency 1σ) is unresolved.
    Nonfinite covariance is not zero error. Numeric fit failures retain a reason
    and trace; malformed source/raw is rejected by the caller before this seam.
    """
    masked = np.where(acquired[..., None], shots, complex(np.nan, np.nan))
    counts = acquired.sum(axis=1) * shots.shape[-1]
    mean = np.full(len(delays), complex(np.nan, np.nan))
    for i in np.flatnonzero(counts):
        mean[i] = np.mean(masked[i][np.isfinite(masked[i])])
    trace = rotate2real(mean).real
    fit = IdleBranchFit(
        dc_index, branch, int(counts.sum()), bool(np.all(acquired)), trace
    )
    if not np.all(acquired):
        return replace(
            fit,
            reason="Required branch is incomplete; all delays and rounds are needed",
        )
    span = float(np.ptp(delays))
    if (
        len(delays) < 6
        or span <= 0
        or not np.allclose(np.diff(delays), np.diff(delays)[0])
    ):
        return replace(
            fit,
            reason="At least six distinct regularly spaced delays with positive span are needed",
        )
    if np.ptp(trace) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(trace)))):
        return replace(fit, reason="No identifiable Ramsey contrast")
    return _fit_complete_branch(fit, delays, detune)


def _fit_complete_branch(
    fit: IdleBranchFit, delays: NDArray[np.float64], detune: float
) -> IdleBranchFit:
    trace = fit.trace
    span = float(np.ptp(delays))
    try:
        _, _, freq, error, curve, (params, covariance) = fit_decay_fringe(delays, trace)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return replace(fit, reason=f"Single-fringe fit failed: {exc}")
    quality = compute_fit_quality(
        trace,
        curve,
        dict(
            zip(
                ("offset", "amplitude", "frequency", "phase", "decay_time"),
                map(float, params),
                strict=True,
            )
        ),
        covariance,
    )
    if not np.isfinite(freq) or not np.all(np.isfinite(curve)):
        return replace(
            fit, reason="Single-fringe fit is nonfinite", fit_quality=quality
        )
    fit = replace(
        fit, fit_curve=curve, magnitude_mhz=abs(float(freq)), fit_quality=quality
    )
    if not np.isfinite(error) or error < 0:
        return replace(
            fit, reason="Frequency uncertainty is not finite; sign is unidentifiable"
        )
    resolution = max(1 / span, float(error))
    magnitude = abs(float(freq))
    reason = None
    if detune == 0 or min(magnitude, abs(magnitude - 2 * abs(detune))) <= resolution:
        reason = "Sign unidentifiable near 0 or 2*abs(detune); possible zero-frequency folding"
    return replace(
        fit,
        error_mhz=float(error),
        resolution_mhz=resolution,
        signed_mhz=None if reason else float(np.sign(detune) * magnitude),
        reason=reason,
    )


def plot_analysis(
    answer: IdleZZAnalysis,
    delays: NDArray[np.float64],
    dc_values: NDArray[np.float64],
    *,
    title: str,
    dc_label: str,
    plots: Plots,
) -> None:
    """Draw available branch traces/fits and ζ without replacing unavailable values.

    delays is the T-point µs axis; dc_values contains the D native coordinates
    or the documented inactive sentinel. answer holds two branches per DC.
    title describes the source target/detune, and dc_label names the saved
    device/unit or inactive axis. plots owns this operation's figures. Register
    ramsey_0..ramsey_{D-1} and zeta, leaving source arrays unchanged. No I/O or
    calibration effects occur; plotting errors propagate to the operation owner.
    """
    for dc_index, dc_value in enumerate(dc_values):
        fig, ax = plots.subplots(f"ramsey_{dc_index}", figsize=(9, 6))
        reasons = []
        for fit in answer.branches[2 * dc_index : 2 * dc_index + 2]:
            label = "I" if fit.branch == 0 else "Xπ"
            ax.plot(
                delays,
                fit.trace,
                ".",
                label=f"Spectator {label} ({fit.acquired_shots} shots)",
            )
            if fit.fit_curve is not None:
                ax.plot(
                    delays,
                    fit.fit_curve,
                    label=f"{label} fit: |f|={fit.magnitude_mhz:.5g} MHz",
                )
            if fit.reason:
                reasons.append(f"{label}: {fit.reason}")
            elif fit.signed_mhz is not None:
                reasons.append(f"{label}: signed f={fit.signed_mhz:.5g} MHz")
        zeta = answer.zeta_mhz[dc_index]
        subtitle = f"ζ={zeta:.5g} MHz" if zeta is not None else "ζ unavailable"
        ax.set(
            xlabel="Delay (us)",
            ylabel="Mean target IQ, principal real axis (a.u.)",
            title=f"{title}; {dc_label}={dc_value:g}\n{subtitle}; single-frequency model estimate\nThermal nominal branches; mixture not excluded, no heralding",
        )
        ax.text(
            0.01,
            -0.25,
            "\n".join(reasons),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    fig, ax = plots.subplots("zeta", figsize=(8, 5))
    for index, (value, error) in enumerate(
        zip(answer.zeta_mhz, answer.zeta_error_mhz, strict=True)
    ):
        if value is not None:
            ax.errorbar(dc_values[index], value, yerr=error, marker="o", color="C0")
    if not any(value is not None for value in answer.zeta_mhz):
        ax.text(
            0.5,
            0.5,
            "ζ unavailable: see branch reasons",
            transform=ax.transAxes,
            ha="center",
        )
    ax.set(
        xlabel=dc_label,
        ylabel="ζ = f_Xπ − f_I (MHz)",
        title="Single-frequency model fit difference\nThermal mixture not excluded; not pure ZZ",
    )
    ax.grid(True)
    fig.tight_layout()
