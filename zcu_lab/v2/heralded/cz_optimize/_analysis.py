"""CZ analysis values, candidate rendering and shot uncertainty."""

from dataclasses import dataclass
from math import degrees

import numpy as np
from numpy.typing import NDArray
from zcu_tools.core.waveform_format import WaveformSamples

from zcu_lab.v2.support.cz_optimize_waveform import Z1, Z2, render_candidate


@dataclass(frozen=True)
class CZParameterValue:
    """One selected physical parameter with a stable name and display unit."""

    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class HeraldedCZOptimizeAnalysis:
    """Best complete evaluation and its physical settings, independent of asset I/O.

    Targets give the mean of gg/ge/eg/ee returns, then the mean across all
    configured seeds. best_class_returns gives the four seed means in that order.
    best_shot_sem is None if any of the selected evaluation's cells has one shot.
    start_target is None without an eligible baseline. best_params follows
    Result's SI column order. For evaluation zero it describes the CMA initial
    mean, not the original pulse. parameters uses display units.
    scored_candidates excludes evaluation zero; missing_cells counts only
    missing classes in measured tables. reason explains unavailable comparison.
    best_raw_gains is None for evaluation zero; otherwise it holds Qubit1 and
    Qubit2 arb gains. The remaining fields describe the saved rendering grid.
    """

    best_evaluation: int
    best_target: float
    start_target: float | None
    best_class_returns: tuple[float, ...]
    best_shot_sem: float | None
    scored_candidates: int
    missing_cells: int
    reason: str | None
    best_params: tuple[float, ...]
    parameters: tuple[CZParameterValue, ...]
    best_raw_gains: tuple[float, float] | None
    start_duration_s: float
    cycle_period_s: float
    samples_per_cycle: int

    def render_best(self) -> WaveformSamples:
        """Render the selected CMA arb, or raise if the original pulse won."""
        if self.best_evaluation == 0:
            raise ValueError("Evaluation 0 plays the original pulse, not a CMA arb")
        return render_candidate(
            np.asarray(self.best_params, dtype=np.float64),
            duration_us=self.start_duration_s * 1e6,
            cycle_us=self.cycle_period_s * 1e6,
            samples_per_cycle=self.samples_per_cycle,
        )

    @property
    def corrections_deg(self) -> tuple[float, float]:
        """Selected Qubit1/Qubit2 local Z settings in degrees."""
        return degrees(self.best_params[Z1]), degrees(self.best_params[Z2])


@dataclass(frozen=True)
class HeraldedCZOptimizeUnavailable:
    """Reason why the record cannot supply a best evaluation."""

    reason: str
    missing_cells: int


@dataclass(frozen=True)
class ReturnObservations:
    """Per-seed measurements and equal-seed, equal-class objectives for full rows."""

    measured: NDArray[np.bool_]
    totals: NDArray[np.float64]
    rates: NDArray[np.float64]
    targets: NDArray[np.float64]
    sem: NDArray[np.float64]


def return_observations(counts: NDArray[np.float64]) -> ReturnObservations:
    """Reduce evaluation × seed × herald × terminal counts (two axes per read).

    The last axis of totals and rates is gg/ge/eg/ee. An unmeasured seed is
    separate from a measured zero denominator. One shot can give a rate but
    not a sample variance of the mean. SEM uses the same fixed weights as
    the score and never estimates seed-to-seed variation.
    """
    q1, q2 = (0, 0, 1, 1), (0, 1, 0, 1)
    measured = np.asarray(np.isfinite(counts).all(axis=(-4, -3, -2, -1)))
    totals = counts[..., q1, q2, :, :].sum(axis=(-2, -1))
    returns = counts[..., q1, q2, q1, q2]
    rates = np.divide(
        returns,
        totals,
        out=np.full_like(totals, np.nan),
        where=measured[..., None] & (totals > 0),
    )
    variance = np.divide(
        rates * (1.0 - rates),
        totals - 1,
        out=np.full_like(totals, np.nan),
        where=totals > 1,
    )
    return ReturnObservations(
        measured=measured,
        totals=totals,
        rates=rates,
        targets=rates.mean(axis=-1).mean(axis=-1),
        sem=np.sqrt(variance.sum(axis=(-2, -1))) / (4 * counts.shape[1]),
    )
