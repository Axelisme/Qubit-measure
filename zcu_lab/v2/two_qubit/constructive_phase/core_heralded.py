"""Ground-ground heralded constructive_phase with saved 81-class counts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
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
from zcu_tools.plotting.plots import Plots

from .._rabi_scan import RabiScanResult
from .analysis import ConstructivePhaseAnalysis
from .core import ConstructivePhaseCfg, acquire_scan, analyze_scan


class ConstructivePhaseHeraldedCfg(ConstructivePhaseCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency. Herald precedes B11 preparation.

    The same full mux cfg supplies herald and terminal reads, separated by
    readout.post_delay. reset=None emits no reset. Other inputs match ordinary
    ConstructivePhaseCfg, including the circle classifiers with radius from GE Post.
    """


@dataclass(frozen=True)
class ConstructivePhaseHeraldedResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3,3,3) counts.

    Class order is g/e/other; axes are herald F1/F2 then terminal F1/F2.
    Unmeasured points are NaN. Counts retain all herald outcomes, unnormalized.
    """


def restore_result(
    data: GroupedLoadData[ConstructivePhaseHeraldedCfg],
) -> ConstructivePhaseHeraldedResult:
    """Restore 81-class sums; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ConstructivePhaseHeraldedResult(
        coordinates=sums.axes[5], durations=sums.axes[4], sums=sums.z.astype(np.float64)
    )


class ConstructivePhaseHeraldedExp(Experiment[ConstructivePhaseHeraldedResult, ConstructivePhaseHeraldedCfg]):
    """Condition constructive_phase analysis on herald classified (g,g)."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ConstructivePhaseHeraldedResult, ConstructivePhaseHeraldedCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec("Drive duration", "s", field_name="durations", scale=1e-6),
                    VariableAxisSpec("F2 relative to F1 phase", "deg", field_name="coordinates", scale=1.0),
                ),
                z=VariableZSpec(field_name="sums", label="Herald and terminal class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=ConstructivePhaseHeraldedResult, cfg_type=ConstructivePhaseHeraldedCfg,
        tag="two_qubit/constructive_phase_heralded", result_builder=restore_result,
    )

    def run(
        self, config: ConstructivePhaseHeraldedCfg, *, context: RunContext
    ) -> ConstructivePhaseHeraldedResult:
        """Acquire optional reset → herald → B11 → drive → double Xπ → terminal.

        Both reads emit one mux object with separate tags. Return all 81-class
        counts, including completed partial rounds. Check context.scope.stop.
        """
        data = acquire_scan(config, context=context, heralded=True)
        return ConstructivePhaseHeraldedResult(data.coordinates, data.durations, data.sums)

    def analyze(
        self, source: RunRecord[ConstructivePhaseHeraldedCfg, ConstructivePhaseHeraldedResult],
        options: None, *, plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Analyze the herald (g,g) slice and plot all nine herald fractions.

        Pass fraction is not ground-state purity. Empty accepted counts give
        NaN terminal probabilities and unresolved fits. Saved cfg is required.
        """
        del options
        if source.cfg is None:
            raise ValueError("Rabi analysis requires saved cfg; current Form/library is not a substitute")
        return analyze_scan(source.cfg, source.result, plots=plots, heralded=True)
