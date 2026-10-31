"""Ground-ground heralded conditional phase."""

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
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .analysis import ConditionalPhaseAnalysis, analyze_phase
from .core import ConditionalPhaseCfg, ConditionalPhaseResult, acquire_counts


class ConditionalPhaseHeraldedCfg(ConditionalPhaseCfg):
    """Conditional phase preceded by optional reset and a mux herald.

    All fields and units follow ConditionalPhaseCfg. Both reads use the same mux
    object and classifiers. readout.post_delay is the wait after herald.
    """


@dataclass(frozen=True)
class ConditionalPhaseHeraldedResult(ConditionalPhaseResult):
    """Scan coordinates follow ConditionalPhaseResult; sums retains all 81 classes.

    sums is float64 (2,2,Nn,Nf,3,3,3,3), ordered herald F1/F2 then terminal
    F1/F2, each g/e/other. No selection or normalization precedes persistence.
    Unmeasured points remain NaN; completed rounds survive cancellation.
    """


def restore_result(
    data: GroupedLoadData[ConditionalPhaseHeraldedCfg],
) -> ConditionalPhaseHeraldedResult:
    """Restore 81-class sums; reject legacy raw-IQ files without sums."""
    sums = data.variable("sums")
    return ConditionalPhaseHeraldedResult(
        branches=sums.axes[7].astype(np.int64),
        quadratures=sums.axes[6].astype(np.int64),
        gate_counts=sums.axes[5].astype(np.int64),
        frequencies=sums.axes[4].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class ConditionalPhaseHeraldedExp(
    Experiment[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
):
    """Analyze classified X/Y projections conditioned on herald (g,g)."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConditionalPhaseHeraldedResult, ConditionalPhaseHeraldedCfg]
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
                        "Common CZ frequency",
                        "Hz",
                        field_name="frequencies",
                        dtype=np.float64,
                        scale=1e6,
                    ),
                    VariableAxisSpec(
                        "CZ gate count", "", field_name="gate_counts", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        "Projection X or Y",
                        "",
                        field_name="quadratures",
                        dtype=np.int64,
                    ),
                    VariableAxisSpec(
                        "Spectator I or Xπ", "", field_name="branches", dtype=np.int64
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
        result_type=ConditionalPhaseHeraldedResult,
        cfg_type=ConditionalPhaseHeraldedCfg,
        tag="two_qubit/conditional_phase_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: ConditionalPhaseHeraldedCfg, *, context: RunContext
    ) -> ConditionalPhaseHeraldedResult:
        """Acquire optional reset → herald → original sequence → terminal sums."""
        return ConditionalPhaseHeraldedResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
            sweep2array(config.frequency),
            acquire_counts(config, context=context, heralded=True),
        )

    def analyze(
        self,
        source: RunRecord[ConditionalPhaseHeraldedCfg, ConditionalPhaseHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConditionalPhaseAnalysis:
        """Use herald (g,g) terminal counts; also show all nine herald fractions."""
        del options
        if source.cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        return analyze_phase(source.cfg, source.result, plots=plots, heralded=True)
