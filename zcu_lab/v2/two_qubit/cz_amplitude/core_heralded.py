"""Common CZ amplitude calibration conditioned on a ground-ground herald."""

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
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .._sums import herald_probabilities, terminal_counts
from .analysis import OUTCOME_LABELS, CZAmplitudeAnalysis, analyze_return
from .core import CZAmplitudeCfg, acquire_counts


class CZAmplitudeHeraldedCfg(CZAmplitudeCfg):
    """Optional reset → herald → B11 → CZ → Xπ⊗Xπ → terminal mux.

    Inherited readout supplies both reads and their g/e/other circle radii.
    Its post_delay is the wait after herald. reset=None emits no reset.
    All other fields and units match CZAmplitudeCfg.
    """


@dataclass(frozen=True)
class CZAmplitudeHeraldedResult:
    """scales is the gain multiplier; sums is float64 (Ns,3,3,3,3) raw counts.

    Class axes are herald F1/F2 then terminal F1/F2, each ordered g/e/other.
    Rounds add without averaging; all herald classes remain stored.
    Unmeasured scales remain NaN; completed rounds survive cancellation.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class CZAmplitudeHeraldedAnalysis:
    """terminal contains the ordinary fit of herald (g,g) terminal counts.

    herald_matrix is (Ns,3,3), ordered g/e/other for F1/F2. pass_fraction is its
    (g,g) cell, shape (Ns,), not ground-state purity. Zero totals give NaN.
    """

    terminal: CZAmplitudeAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[CZAmplitudeHeraldedCfg],
) -> CZAmplitudeHeraldedResult:
    """Restore 81-class sums; old per-shot IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return CZAmplitudeHeraldedResult(sums.axes[4], sums.z.astype(np.float64))


class CZAmplitudeHeraldedExp(
    Experiment[CZAmplitudeHeraldedResult, CZAmplitudeHeraldedCfg]
):
    """Scan common gain scales with ground-ground herald post-selection."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[CZAmplitudeHeraldedResult, CZAmplitudeHeraldedCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec("Common gain scale", "", field_name="scales"),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Herald and terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=CZAmplitudeHeraldedResult,
        cfg_type=CZAmplitudeHeraldedCfg,
        tag="two_qubit/cz_amplitude_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: CZAmplitudeHeraldedCfg, *, context: RunContext
    ) -> CZAmplitudeHeraldedResult:
        """Acquire unchanged raw sums for all 81 herald/terminal outcomes.

        No host round loop or per-shot IQ is stored. Check the RunContext stop
        signal for acquisition errors before interpreting partial results.
        """
        return CZAmplitudeHeraldedResult(
            sweep2array(config.scales),
            acquire_counts(config, context=context, heralded=True),
        )

    def analyze(
        self,
        source: RunRecord[CZAmplitudeHeraldedCfg, CZAmplitudeHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> CZAmplitudeHeraldedAnalysis:
        """Fit accepted terminal P00 and plot all nine herald probabilities.

        Fit and coverage policy match the ordinary version, using all herald
        classes to determine acquisition completeness. No accepted shots gives
        NaN terminal probabilities and no proposal. Missing cfg raises ValueError.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("CZ return analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scales)):
            raise ValueError("CZ return scales must match saved cfg")
        if result.sums.shape != (len(result.scales), 3, 3, 3, 3):
            raise ValueError("Heralded CZ return requires sums (Ns,3,3,3,3)")
        terminal = analyze_return(
            result.scales,
            terminal_counts(result.sums, heralded=True),
            acquired_shots=result.sums.sum(axis=(-4, -3, -2, -1)),
            expected_shots=cfg.reps * cfg.rounds,
            plots=plots,
        )
        herald_matrix = herald_probabilities(result.sums)
        _, ax = plots.subplots("herald", figsize=(10, 5), layout="constrained")
        for index, label in enumerate(OUTCOME_LABELS):
            ax.plot(
                result.scales, herald_matrix.reshape(-1, 9)[:, index], ".-", label=label
            )
        ax.set(
            xlabel="Common gain scale",
            ylabel="Herald probability",
            title="Herald (g,g) is pass fraction, not ground-state purity",
        )
        ax.legend()
        return CZAmplitudeHeraldedAnalysis(
            terminal, herald_matrix, herald_matrix[:, 0, 0]
        )
