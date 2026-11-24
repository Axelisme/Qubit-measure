"""Four-input CZ return conditioned on a measured pre-CZ herald."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, Self

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Reset, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support.cz_return import CZReturnFit, fit_return
from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    joint_probabilities,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg

_HERALDS = ("gg", "ge", "eg", "ee")
_CZ_INPUTS = ("11", "10", "01", "00")
_SCALE_LABEL = "Common CZ gain multiplier (dimensionless)"


class HeraldedCZAmplitudeCfg(ConfigBase):
    """Four nominal preparations followed by a visible herald, double Xπ, CZ,
    double Xπ and terminal mux. Each of the two tone fields is a zero-based
    position in the full mux tone list, not an ADC channel number. The same
    calibrated readout is used twice. Reset is ordinary only (None omits it).
    Signed drive gains share one increasing nonnegative dimensionless scale.
    Reps are attempts per preparation and scale per round.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an ordinary reset before nominal preparation, or leave blank.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 Xπ for nominal preparation and both inversion pairs.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 Xπ for nominal preparation and both inversion pairs.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-line CZ drive",
            description="Simultaneous CZ pulses on the two qubit charge lines. Multiply their signed source gains by the common sweep multiplier; keep their relative phase fixed.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full-group mux readout with GE Post radii and the wait after each read. "
            + POINT_ALIAS_DESCRIPTION,
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    scale: Annotated[
        SweepCfg,
        FieldMeta(
            label="Common CZ gain multiplier",
            unit="a.u.",
            description="Multiply both signed CZ gains by each value; 0 omits drive and 1 uses the source gains. Include 0 and scan through the first positive return.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Shots attempted for each nominal preparation and gain point in each round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of complete gain sweeps whose class counts are added.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require calibrated tone positions, matching drive lines and a valid scan."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each CZ line must use its qubit's calibrated Xπ generator")
        scales = sweep2array(self.scale)
        if (
            len(scales) < 2
            or not np.all(np.isfinite(scales))
            or np.any(scales < 0)
            or np.any(np.diff(scales) <= 0)
        ):
            raise ValueError("Common gain scales must be finite, nonnegative and strictly increasing")
        return self


@dataclass(frozen=True)
class HeraldedCZAmplitudeResult:
    """Dimensionless scales and float64 counts from completed rounds.

    sums has axes (prepared Qubit1, prepared Qubit2, scale, herald Qubit1,
    herald Qubit2, terminal Qubit1, terminal Qubit2). Preparations are I, Xπ;
    each classification is g, e, Other. NaN means unmeasured, not zero counts.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class HeraldedCZAmplitudeAnalysis:
    """First-return diagnostics from the measured gg herald row only.

    scale is the first positive return covered by measurements starting at
    zero, or None with a reason. period is the fitted period (dimensionless)
    and error its standard error. offset and contrast describe conditional
    P(gg). fit_quality reports residual and uncertainty metrics without an
    acceptance threshold. Numeric diagnostics are None if unavailable.
    """

    scale: float | None
    period: float | None
    error: float | None
    offset: float | None
    contrast: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class _Returns:
    probability: NDArray[np.float64]
    sem: NDArray[np.float64]
    shots: NDArray[np.float64]


class HeraldedCZAmplitudeExp(
    Experiment[HeraldedCZAmplitudeResult, HeraldedCZAmplitudeCfg]
):
    """Acquire and analyze the four measured CZ return branches."""

    def analyze(
        self,
        source: RunRecord[HeraldedCZAmplitudeCfg, HeraldedCZAmplitudeResult],
        options: None,
        *,
        plots: Plots,
    ) -> HeraldedCZAmplitudeAnalysis:
        """Plot four measured conditional returns and fit only the gg row.

        Use the saved cfg and pool both nominal preparation axes before
        normalizing the rows. A zero or unmeasured gg point ends the fit prefix;
        missing spectator points do not. Missing cfg, inconsistent scan axes
        and invalid counts raise ValueError. No writeback or reclassification.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("Heralded CZ amplitude analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scale)):
            raise ValueError("Heralded CZ amplitude scales must match saved cfg")
        if result.sums.shape != (2, 2, len(result.scales), 3, 3, 3, 3):
            raise ValueError("Heralded CZ amplitude requires sums (2,2,scale,3,3,3,3)")

        returns = _conditional_returns(result.sums)
        gg_shots = returns.shots[:, 0, 0]
        fit = fit_return(
            result.scales,
            returns.probability[:, 0, 0],
            np.isfinite(gg_shots) & (gg_shots > 0),
            gg_shots,
        )
        _plot_returns(plots, result.scales, returns, fit)
        return HeraldedCZAmplitudeAnalysis(
            fit.scale, fit.period, fit.error, fit.offset, fit.contrast, fit.fit_quality, fit.reason
        )


def _conditional_returns(sums: NDArray[np.float64]) -> _Returns:
    """Pool preparations; calculate P(terminal = herald) and its shot SEM."""
    terminal = sums[:, :, :, :2, :2].sum(axis=(0, 1))
    shots = terminal.sum(axis=(-2, -1))
    probabilities = joint_probabilities(terminal)
    covariance = class_probability_covariance(terminal.reshape(-1, 2, 2, 9))
    q1, q2 = np.indices((2, 2))
    hit = 3 * q1 + q2
    return _Returns(
        probabilities[:, q1, q2, q1, q2],
        np.sqrt(covariance[:, q1, q2, hit, hit]),
        shots,
    )


def _return_figure(plots: Plots, name: str) -> Figure:
    """Label four return panels by measured herald and inferred CZ input."""
    fig, _ = plots.subplots(
        num=name, nrows=2, ncols=2, sharex=True, sharey=True,
        figsize=(10, 7), layout="constrained",
    )
    fig.supxlabel(_SCALE_LABEL)
    fig.supylabel("P(terminal = herald | measured herald)")
    for ax, herald, ket in zip(fig.axes, _HERALDS, _CZ_INPUTS, strict=True):
        ax.set(title=f"Herald {herald} → inferred CZ input |{ket}⟩", ylim=(-0.04, 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig


def _plot_returns(
    plots: Plots,
    scales: NDArray[np.float64],
    returns: _Returns,
    fit: CZReturnFit,
) -> None:
    """Draw four conditional traces with shot SEM and the gg-only fit overlay."""
    fig = _return_figure(plots, "heralded_return")
    for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
        ax.errorbar(
            scales,
            returns.probability[:, q1, q2],
            yerr=returns.sem[:, q1, q2],
            fmt=".-",
            label="Measured ± shot SEM",
        )

    gg = fig.axes[0]
    fitted = np.isfinite(fit.curve)
    if fitted.any():
        gg.plot(scales[fitted], fit.curve[fitted], "k-", label="Fixed-phase cosine fit")
    if fit.scale is None:
        diagnostic = fill(f"No gg first return: {fit.reason}", width=85)
    else:
        diagnostic = f"gg first return = {fit.scale:.5g} ± {fit.error:.2g}"
        gg.axvline(fit.scale, color="C3", ls="--", label="Selected first return")
    gg.legend()
    fig.suptitle(f"Heralded CZ amplitude · Qubit1, Qubit2\n{diagnostic}")
