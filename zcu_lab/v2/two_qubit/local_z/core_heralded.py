"""Ground-ground heralded two-target local Z."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from zcu_tools.experiment import (
    Experiment, GroupedAxesSpec, GroupedLoadData,
    VariableAxisSpec, VariableSpec, VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots

from .analysis import LocalZAnalysis, analyze_local_z
from .core import LocalZCfg, LocalZResult, acquire_counts


class LocalZHeraldedCfg(LocalZCfg):
    """Both local-Z targets preceded by optional reset and a mux herald.

    All fields and units follow LocalZCfg. Both reads use the same mux
    object and classifiers. readout.post_delay is the wait after herald.
    """


@dataclass(frozen=True)
class LocalZHeraldedResult(LocalZResult):
    """Scan coordinates follow LocalZResult; sums retains all 81 classes.

    sums is float64 (2,2,Nn,3,3,3,3), ordered herald F1/F2 then terminal
    F1/F2, each g/e/other. No selection or normalization precedes persistence.
    Unmeasured points remain NaN; completed rounds survive cancellation.
    """


def restore_result(
    data: GroupedLoadData[LocalZHeraldedCfg],
) -> LocalZHeraldedResult:
    """Restore 81-class sums; reject legacy raw-IQ files without sums."""
    sums = data.variable("sums")
    return LocalZHeraldedResult(
        targets=sums.axes[6].astype(np.int64),
        quadratures=sums.axes[5].astype(np.int64),
        gate_counts=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class LocalZHeraldedExp(Experiment[LocalZHeraldedResult, LocalZHeraldedCfg]):
    """Analyze classified X/Y projections conditioned on herald (g,g)."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZHeraldedResult, LocalZHeraldedCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec("CZ gate count", "", field_name="gate_counts", dtype=np.int64),
                        VariableAxisSpec("Projection X or Y", "", field_name="quadratures", dtype=np.int64),
                        VariableAxisSpec("Target F1 or F2", "", field_name="targets", dtype=np.int64),
                ),
                z=VariableZSpec(field_name="sums", label="Herald and terminal class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=LocalZHeraldedResult,
        cfg_type=LocalZHeraldedCfg,
        tag="two_qubit/local_z_heralded",
        result_builder=restore_result,
    )

    def run(self, config: LocalZHeraldedCfg, *, context: RunContext) -> LocalZHeraldedResult:
        """Acquire optional reset → herald → original sequence → terminal sums."""
        return LocalZHeraldedResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
            acquire_counts(config, context=context, heralded=True),
        )

    def analyze(
        self, source: RunRecord[LocalZHeraldedCfg, LocalZHeraldedResult],
        options: None, *, plots: Plots,
    ) -> LocalZAnalysis:
        """Use herald (g,g) terminal counts; also show all nine herald fractions."""
        del options
        if source.cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        return analyze_local_z(source.cfg, source.result, plots=plots, heralded=True)
