"""Shared raw duration scans for constructive phase and Rabi chevron only."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.analysis.fitting import compute_fit_quality
from zcu_tools.analysis.fitting.quality import FitQuality
from zcu_tools.analysis.fitting.rabi import fit_rabi
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg

from zcu_lab.v2.support import live_line

from ._drive import DualLineCfg, dual_line_drive
from ._readout import JointRoles, collect_joint_shots
from .joint_readout.core import GECenters


class DurationScanCfg(ConfigBase):
    """Shared B11 → two-line drive → double Xπ → mux inputs.

    duration is the requested positive envelope length in us on both lines.
    drive carries the source pair: common MHz frequency, signed gains, phases
    in degrees, explicit waveforms and timing. Only scalar length-settable
    envelopes are supported; arbitrary assets are not resized. roles and GE
    centers are saved capture values, never live lookup. Every shot is kept.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    x180_f1: Annotated[
        PulseCfg,
        FieldMeta(
            label="F1 Xπ",
            description="填入 F1 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    x180_f2: Annotated[
        PulseCfg,
        FieldMeta(
            label="F2 Xπ",
            description="填入 F2 x180 slot，先做 amplitude／length Rabi。frequency MHz、gain a.u. within [-1, 1]、duration us。",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="填入 full-group mux readout slot，先用 lookback 取得 timing us，再做 mux onetone。",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="填入 F1/F2 charge-line pulses，先以 conditional spectroscopy 選頻，再確認 cz_f1／cz_f2 slots。",
        ),
    ]
    duration: Annotated[
        SweepCfg,
        FieldMeta(
            unit="us",
            label="Duration",
            description="填入兩條 charge-line 的共同 duration 起訖值，單位 us。先做 conditional spectroscopy 選頻，再由操作者確認 chevron 時間範圍。",
        ),
    ]
    f1_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F1 GE centers",
            description="填入 singleshot/ge 取得的 F1 normalized complex IQ centers。",
        ),
    ]
    f2_centers: Annotated[
        GECenters,
        FieldMeta(
            label="F2 GE centers",
            description="填入 singleshot/ge 取得的 F2 normalized complex IQ centers。",
        ),
    ]

    @model_validator(mode="after")
    def validate_duration_scan(self) -> Self:
        """Reject invalid tones, routes, durations and unscalable waveforms."""
        self.roles.validate_readout(self.readout)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each drive must use its role's calibrated Xπ generator")
        durations = sweep2array(self.duration)
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("Drive durations must be finite and positive")
        for pulse in (self.drive.f1, self.drive.f2):
            probe = pulse.model_copy(deep=True)
            probe.set_param("length", float(durations[0]))
        return self


@dataclass(frozen=True)
class RabiScanResult:
    """Raw (coordinate,duration,rounds*reps,N) complex IQ in original tone order.

    coordinates are relative phase degrees or common MHz frequency, according
    to the leaf's native AxesSpec. durations are requested us. shot_indices are
    round-major, tone_indices are original full mux indices. Unwritten whole
    rounds remain complex NaN; acquired nonfinite IQ is an error, not a sentinel.
    """

    coordinates: NDArray[np.float64]
    durations: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


def run_duration_scan(
    config: DurationScanCfg,
    coordinates: NDArray[np.float64],
    drives: list[DualLineCfg],
    *,
    context: RunContext,
) -> RabiScanResult:
    """Acquire one explicit drive per coordinate, then scan common duration.

    Leaf callers supply finite coordinates and equally many validated pairs.
    Schedule writes fixed host-round slots, including final overwrite; errors
    stay in context.scope.stop and the operation owner must raise_if_error().
    Inputs are not mutated. No library, hardware discovery or pulse optimization.
    """
    if len(drives) != len(coordinates) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Each finite coordinate requires one explicit drive pair")
    cfg = config.model_copy(deep=True, update={"rounds": 1})
    durations = sweep2array(cfg.duration)
    ntones = len(cfg.readout.tones)
    viewer = live_line(
        context.plots,
        "measurement",
        "Drive duration (us)",
        "Completed shots (all coordinates)",
        num_lines=1,
    )

    def update_view(raw: NDArray[np.complex128]) -> None:
        viewer.update(durations, np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 2, 3)))

    buffer = SignalBuffer(
        (len(drives), len(durations), config.rounds, cfg.reps, ntones),
        on_update=update_view,
    )
    with Schedule(cfg, buffer, scope=context.scope) as sched:
        for index, step in sched.scan("coordinate", list(range(len(drives)))):
            for duration, duration_step in step.scan("duration", durations.tolist()):
                drive = drives[index].model_copy(deep=True)
                for pulse in (drive.f1, drive.f2):
                    pulse.set_param("length", duration)
                builder = duration_step.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=duration_step.cfg.reps,
                    initial_delay=duration_step.cfg.initial_delay,
                    relax_delay=duration_step.cfg.relax_delay,
                )
                m = cfg
                builder.add_pulse("prepare_f1", m.x180_f1)
                builder.add_pulse("prepare_f2", m.x180_f2)
                builder.add(dual_line_drive(drive, frequency=float(drive.f1.freq)))
                builder.add_pulse("contrast_f1", m.x180_f1)
                builder.add_pulse("contrast_f2", m.x180_f2)
                builder.add(MuxReadout("readout", m.readout))
                prog = builder.build()
                for _, round_step in duration_step.repeat("round", config.rounds):
                    _ = round_step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=round_step.cfg.reps,
                        initial_delay=round_step.cfg.initial_delay,
                        relax_delay=round_step.cfg.relax_delay,
                    ).run_program(
                        prog,
                        raw2signal_fn=partial(collect_joint_shots, prog),
                        rounds=round_step.cfg.rounds,
                    )
    buffer.trigger_update(flush=True)
    return RabiScanResult(
        coordinates,
        durations,
        np.arange(config.rounds * cfg.reps, dtype=np.int64),
        np.arange(ntones, dtype=np.int64),
        buffer.array.reshape(
            len(drives), len(durations), config.rounds * cfg.reps, ntones
        ),
    )


@dataclass(frozen=True)
class ReturnData:
    """Complete-point same-shot joint probabilities and diagnostic IQ.

    populations is (coordinate,duration,4), columns 00/01/10/11. mean_iq is
    (coordinate,duration,N). Incomplete points stay NaN. complete and acquired_shots
    are (coordinate,duration); all expected rounds are necessary to analyze a
    point. No radius cut, matrix inversion or independent marginal product.
    """

    populations: NDArray[np.float64]
    mean_iq: NDArray[np.complex128]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.int64]


def joint_return(
    cfg: DurationScanCfg, data: RabiScanResult, coordinates: NDArray[np.float64]
) -> ReturnData:
    """Validate saved cfg/raw and classify complete duration points, keeping all shots.

    Require source axes (within SI conversion roundoff) and whole finite or whole NaN round slots. Invalid
    acquired IQ or partial missing rounds raise ValueError. Missing GE centers
    are rejected by cfg validation, never reconstructed from a current library.
    """
    durations = sweep2array(cfg.duration)
    nc, nd = len(coordinates), len(durations)
    nt, ns = len(cfg.readout.tones), cfg.rounds * cfg.reps
    if (
        data.coordinates.shape != coordinates.shape
        or not np.allclose(
            data.coordinates, coordinates, rtol=8 * np.finfo(float).eps, atol=0
        )
        or data.durations.shape != durations.shape
        or not np.allclose(
            data.durations, durations, rtol=8 * np.finfo(float).eps, atol=0
        )
        or not np.array_equal(data.shot_indices, np.arange(ns))
        or not np.array_equal(data.tone_indices, np.arange(nt))
        or data.signals.shape != (nc, nd, ns, nt)
    ):
        raise ValueError("Rabi raw axes/shape must match saved cfg and full tones")
    rounds = data.signals.reshape(nc, nd, cfg.rounds, cfg.reps, nt)
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError(
            "Rabi raw contains nonfinite acquired IQ or a partially missing round"
        )
    complete = np.all(acquired, axis=-1)
    populations = np.full((nc, nd, 4), np.nan)
    mean_iq = np.full((nc, nd, nt), complex(np.nan, np.nan), dtype=np.complex128)
    for c, d in np.argwhere(complete):
        shots = data.signals[c, d]
        outcomes = np.zeros(ns, dtype=np.int64)
        for tone, centers, weight in (
            (cfg.roles.f1_tone, cfg.f1_centers, 2),
            (cfg.roles.f2_tone, cfg.f2_centers, 1),
        ):
            values = shots[:, tone]
            outcomes += weight * (
                np.abs(values - centers.e) < np.abs(values - centers.g)
            )
        populations[c, d] = np.bincount(outcomes, minlength=4) / ns
        mean_iq[c, d] = shots.mean(axis=0)
    return ReturnData(
        populations, mean_iq, complete, acquired.sum(axis=-1, dtype=np.int64) * cfg.reps
    )


def plot_return_data(
    data: RabiScanResult,
    answer: ReturnData,
    *,
    coordinate_label: str,
    map_name: str,
    plots: Plots,
) -> None:
    """Draw joint return map, all outcome curves and complete-point full-tone I/Q.

    map_name is the caller's figure name (interference or chevron), not a model
    selector. No calibration, gate selection or fit is performed here.
    """
    fig, ax = plots.subplots(map_name, figsize=(9, 6), layout="constrained")
    image = ax.pcolormesh(
        data.durations,
        data.coordinates,
        answer.populations[..., 0],
        shading="nearest",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(image, ax=ax, label="Observed joint P00 after double Xπ")
    ax.set(
        xlabel="Drive duration (us)",
        ylabel=coordinate_label,
        title="Thermal nominal B11 return proxy · not gate fidelity",
    )
    fig, _ = plots.subplots(
        "return", nrows=2, ncols=2, figsize=(11, 7), squeeze=False, layout="constrained"
    )
    for outcome, ax in enumerate(fig.axes):
        for coordinate, values in zip(
            data.coordinates, answer.populations, strict=True
        ):
            ax.plot(data.durations, values[:, outcome], ".-", label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel=f"Observed P{outcome:02b}",
            title=f"Joint {outcome:02b} · all classified shots",
        )
        if len(data.coordinates) <= 25:
            ax.legend(title=coordinate_label, fontsize="x-small", ncols=2)
    for tone in data.tone_indices:
        fig, _ = plots.subplots(
            f"iq_tone_{tone}", nrows=2, figsize=(10, 7), layout="constrained"
        )
        for ax, values, label in (
            (fig.axes[0], answer.mean_iq.real, "I"),
            (fig.axes[1], answer.mean_iq.imag, "Q"),
        ):
            image = ax.pcolormesh(
                data.durations, data.coordinates, values[..., tone], shading="nearest"
            )
            fig.colorbar(image, ax=ax, label=f"Mean {label} (ADC units/sample)")
            ax.set(
                xlabel="Drive duration (us)",
                ylabel=coordinate_label,
                title=f"Original tone {tone} · {label}",
            )


@dataclass(frozen=True)
class RabiRate:
    """One complete duration-slice cosine fit in MHz, or an explicit failure.

    rate is None when unresolved; reason explains why. uncertainty is the finite
    frequency standard error, not an acceptance threshold. fitted contains model
    values at all saved durations, or None on failure. quality reports the actual
    fit observations/parameters/covariance, including rejected low-rate fits.
    """

    rate: float | None
    uncertainty: float | None
    reason: str | None
    fitted: NDArray[np.float64] | None
    quality: FitQuality | None


@dataclass(frozen=True)
class RabiAnalysis:
    """Same-shot joint return data and one duration-slice fit per coordinate."""

    return_data: ReturnData
    rates: tuple[RabiRate, ...]


def fit_duration_rates(
    data: RabiScanResult, answer: ReturnData
) -> tuple[RabiRate, ...]:
    """Fit complete regular duration slices with the existing undamped cosine.

    A fitted rate below 1/actual saved duration span is unresolved. No automatic
    Nyquist, contrast/sigma, frequency/sigma, chi-square or fidelity gate is used.
    Numerical failure, zero contrast or nonfinite uncertainty produces a reason,
    not a zero rate or optimizer fallback. No incomplete trace is fitted.
    """
    return tuple(
        _fit_duration_rate(data.durations, values[:, 0], complete=bool(complete.all()))
        for values, complete in zip(answer.populations, answer.complete, strict=True)
    )


def _duration_fit_reason(
    durations: NDArray[np.float64], values: NDArray[np.float64], *, complete: bool
) -> str | None:
    if not complete:
        return "Incomplete duration trace"
    if len(durations) <= 4 or np.ptp(durations) == 0:
        return "Need more than four distinct duration samples"
    if not np.allclose(np.diff(durations), durations[1] - durations[0]):
        return "Cosine fit requires regularly spaced durations"
    if np.ptp(values) == 0:
        return "Zero return contrast: Rabi rate undefined"
    return None


def _fit_duration_rate(
    durations: NDArray[np.float64], values: NDArray[np.float64], *, complete: bool
) -> RabiRate:
    reason = _duration_fit_reason(durations, values, complete=complete)
    if reason:
        return RabiRate(None, None, reason, None, None)
    try:
        _, _, _, _, frequency, error, fitted, (parameters, covariance) = fit_rabi(
            durations, values, decay=False
        )
    except (
        ValueError,
        RuntimeError,
        ZeroDivisionError,
        FloatingPointError,
        OverflowError,
    ) as exc:
        return RabiRate(None, None, f"Cosine fit failed: {exc}", None, None)
    quality = compute_fit_quality(
        values,
        fitted,
        dict(
            zip(
                ("offset", "contrast", "frequency", "phase_deg"),
                parameters,
                strict=True,
            )
        ),
        covariance,
    )
    if (
        not np.all(np.isfinite(parameters))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(fitted))
        or np.any(np.diag(covariance) < 0)
        or not np.isfinite(error)
    ):
        return RabiRate(
            None, None, "Nonfinite fit or unavailable uncertainty", None, quality
        )
    if parameters[1] == 0 or frequency == 0:
        return RabiRate(
            None,
            None,
            "Degenerate cosine: contrast or frequency is zero",
            None,
            quality,
        )
    if abs(frequency) < 1 / np.ptp(durations):
        return RabiRate(
            None, None, "Rate below 1/duration span: unresolved", fitted, quality
        )
    return RabiRate(abs(frequency), error, None, fitted, quality)


def plot_duration_rates(
    data: RabiScanResult,
    rates: tuple[RabiRate, ...],
    *,
    coordinate_label: str,
    plots: Plots,
) -> None:
    """Plot valid rates with uncertainty and measured/fitted duration traces."""
    _, ax = plots.subplots("rabi_rate", figsize=(10, 6), layout="constrained")
    for coordinate, rate in zip(data.coordinates, rates, strict=True):
        if rate.rate is not None:
            ax.errorbar(
                coordinate, rate.rate, yerr=rate.uncertainty, fmt="o", color="C0"
            )
    reasons = [
        f"{coordinate:g}: {rate.reason}"
        for coordinate, rate in zip(data.coordinates, rates, strict=True)
        if rate.reason
    ]
    if reasons:
        ax.text(
            0.02,
            0.98,
            "\n".join(reasons),
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
    ax.set(
        xlabel=coordinate_label,
        ylabel="Rabi rate (MHz)",
        title="Undamped cosine model · rates below 1/duration span excluded",
    )
    _, ax = plots.subplots("rabi_traces", figsize=(10, 6), layout="constrained")
    for coordinate, rate in zip(data.coordinates, rates, strict=True):
        if rate.fitted is not None:
            ax.plot(data.durations, rate.fitted, label=f"{coordinate:g}")
    ax.set(
        xlabel="Drive duration (us)",
        ylabel="Fitted joint P00",
        title="Duration cosine fits · inspect model against measured return curves",
    )
    if ax.lines and len(data.coordinates) <= 25:
        ax.legend(title=coordinate_label, fontsize="x-small", ncols=2)
