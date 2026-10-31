"""Joint readout with a ground-ground herald before each preparation."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
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

from .._sums import herald_probabilities, terminal_counts
from .core import (
    BRANCH_LABELS,
    OUTCOME_LABELS,
    JointReadoutAnalysis,
    JointReadoutCfg,
    JointReadoutResult,
    acquire_counts,
    analyze_counts,
)


class JointReadoutHeraldedCfg(JointReadoutCfg):
    """Four branches preceded by optional reset and a full mux herald read.

    Inherited readout supplies both reads and their g/e/other circle radii.
    Its post_delay is the wait after herald. reset=None emits no reset.
    All other fields and units match JointReadoutCfg.
    """


@dataclass(frozen=True)
class JointReadoutHeraldedResult:
    """branches is int64 [0,1,2,3]; sums is float64 (4,3,3,3,3) raw counts.

    Class axes are herald F1, herald F2, terminal F1, terminal F2. Each is ordered
    g/e/other. Rounds add without averaging. Unmeasured branches remain NaN;
    completed rounds survive cancellation. Check the RunContext stop signal for
    acquisition errors.
    """

    branches: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class JointReadoutHeraldedAnalysis:
    """terminal is the 4×9 analysis conditioned on herald classified (g,g).

    herald_matrix is (4,3,3), the herald g/e/other distribution per branch.
    pass_fraction is its (g,g) cell, shape (4,), not ground-state purity.
    Empty or unmeasured distributions are NaN.
    """

    terminal: JointReadoutAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[JointReadoutHeraldedCfg],
) -> JointReadoutHeraldedResult:
    """Restore 81-class sums; missing sums in old raw-IQ files raises ValueError."""
    sums = data.variable("sums")
    return JointReadoutHeraldedResult(
        branches=sums.axes[4].astype(np.int64), sums=sums.z.astype(np.float64)
    )


class JointReadoutHeraldedExp(
    Experiment[JointReadoutHeraldedResult, JointReadoutHeraldedCfg]
):
    """Condition four-branch readout analysis on herald classified (g,g)."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[JointReadoutHeraldedResult, JointReadoutHeraldedCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec(
                        "Nominal preparation", "", field_name="branches", dtype=np.int64
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Herald and terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=JointReadoutHeraldedResult,
        cfg_type=JointReadoutHeraldedCfg,
        tag="two_qubit/joint_readout_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: JointReadoutHeraldedCfg, *, context: RunContext
    ) -> JointReadoutHeraldedResult:
        """Acquire optional reset → herald → I/Xπ preparation → terminal.

        Both tagged reads emit the same mux object. Return (4,3,3,3,3) raw
        summed counts for every herald class, not only accepted shots. The cfg
        is unchanged. Rounds accumulate on the acquisition side.
        """
        return JointReadoutHeraldedResult(
            np.arange(4, dtype=np.int64),
            acquire_counts(config, context=context, heralded=True),
        )

    def analyze(
        self,
        source: RunRecord[JointReadoutHeraldedCfg, JointReadoutHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> JointReadoutHeraldedAnalysis:
        """Analyze herald (g,g) terminal counts and plot all nine herald fractions.

        The terminal matrix uses the ordinary version's analysis. Pass fraction
        measures classified acceptance, not initial-state purity. No cfg or raw
        IQ is needed. Zero acceptance gives NaN terminal probabilities.
        """
        del options
        result = source.result
        if result.sums.shape != (4, 3, 3, 3, 3):
            raise ValueError("Heralded joint readout requires sums (4,3,3,3,3)")
        terminal = analyze_counts(
            JointReadoutResult(
                result.branches, terminal_counts(result.sums, heralded=True)
            ),
            plots=plots,
        )
        herald_matrix = herald_probabilities(result.sums)
        fig, axes = plots.subplots("herald", figsize=(12, 5))
        image = axes.imshow(
            np.ma.masked_invalid(herald_matrix.reshape(4, 9)),
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        axes.set(
            xticks=range(9),
            xticklabels=OUTCOME_LABELS,
            yticks=range(4),
            yticklabels=BRANCH_LABELS,
            xlabel="Herald class (F1, F2)",
            ylabel="Nominal I/Xπ preparation",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=axes, label="Herald probability")
        fig.tight_layout()
        return JointReadoutHeraldedAnalysis(
            terminal, herald_matrix, herald_matrix[..., 0, 0]
        )
