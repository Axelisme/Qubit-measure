"""Ground-ground heralded chevron with saved 81-class counts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from zcu_tools.experiment import Experiment, GroupedAxesSpec, GroupedLoadData, VariableAxisSpec, VariableSpec, VariableZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots

from .._rabi_scan import RabiScanResult
from .._rabi_analysis import RabiAnalysis
from .core import ChevronCfg, acquire_scan, analyze_scan


class ChevronHeraldedCfg(ChevronCfg):
    """Common MHz frequency scan with fixed signed gains and relative phase. Herald precedes B11 preparation.

    The same full mux cfg supplies herald and terminal reads, separated by
    readout.post_delay. reset=None emits no reset. Other inputs match ordinary
    ChevronCfg, including the circle classifiers with radius from GE Post.
    """


@dataclass(frozen=True)
class ChevronHeraldedResult(RabiScanResult):
    """Common drive frequency coordinates, durations in us and (Nc,Nd,3,3,3,3) counts.

    Class order is g/e/other; axes are herald F1/F2 then terminal F1/F2.
    Unmeasured points are NaN. Counts retain all herald outcomes, unnormalized.
    """


def restore_result(
    data: GroupedLoadData[ChevronHeraldedCfg],
) -> ChevronHeraldedResult:
    """Restore 81-class sums; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ChevronHeraldedResult(
        coordinates=sums.axes[5], durations=sums.axes[4], sums=sums.z.astype(np.float64)
    )


class ChevronHeraldedExp(Experiment[ChevronHeraldedResult, ChevronHeraldedCfg]):
    """Condition chevron analysis on herald classified (g,g)."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ChevronHeraldedResult, ChevronHeraldedCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec("Drive duration", "s", field_name="durations", scale=1e-6),
                    VariableAxisSpec("Common drive frequency", "Hz", field_name="coordinates", scale=1e6),
                ),
                z=VariableZSpec(field_name="sums", label="Herald and terminal class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=ChevronHeraldedResult, cfg_type=ChevronHeraldedCfg,
        tag="two_qubit/chevron_heralded", result_builder=restore_result,
    )

    def run(
        self, config: ChevronHeraldedCfg, *, context: RunContext
    ) -> ChevronHeraldedResult:
        """Acquire optional reset → herald → B11 → drive → double Xπ → terminal.

        Both reads emit one mux object with separate tags. Return all 81-class
        counts, including completed partial rounds. Check context.scope.stop.
        """
        data = acquire_scan(config, context=context, heralded=True)
        return ChevronHeraldedResult(data.coordinates, data.durations, data.sums)

    def analyze(
        self, source: RunRecord[ChevronHeraldedCfg, ChevronHeraldedResult],
        options: None, *, plots: Plots,
    ) -> RabiAnalysis:
        """Analyze the herald (g,g) slice and plot all nine herald fractions.

        Pass fraction is not ground-state purity. Empty accepted counts give
        NaN terminal probabilities and unresolved fits. Saved cfg is required.
        """
        del options
        if source.cfg is None:
            raise ValueError("Rabi analysis requires saved cfg; current Form/library is not a substitute")
        return analyze_scan(source.cfg, source.result, plots=plots, heralded=True)
