"""CZ analysis values, candidate rendering and shot uncertainty."""

from dataclasses import dataclass
from math import degrees

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.core.waveform_format import WaveformSamples

from zcu_lab.v2.support.cz_optimize_waveform import Z1, Z2, render_candidate


@dataclass(frozen=True)
class CZParameterValue:
    """One selected physical parameter with a stable name and display unit."""

    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class CZOptimizeAnalysis:
    """Best complete evaluation and its physical settings, independent of asset I/O.

    best_target/start_target are mean retained-seed P(g,g); start_target is
    None when the baseline was not measured. best_params follows Result's SI
    column order. For evaluation zero it describes the CMA initial mean, not
    the original pulse. parameters uses display units.
    best_raw_gains is None for evaluation zero; otherwise it holds Qubit1 and
    Qubit2 arb gains. The remaining fields describe the saved rendering grid.
    """

    best_evaluation: int
    best_target: float
    start_target: float | None
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
class CZOptimizeUnavailable:
    """Reason why the record cannot supply a best evaluation."""

    reason: str


def objective_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Propagate retained-shot P(g,g) SEMs through an equal seed mean."""
    covariance = class_probability_covariance(counts.reshape(*counts.shape[:-2], 9))
    return np.sqrt(covariance[..., 0, 0].sum(axis=-1)) / counts.shape[-3]
