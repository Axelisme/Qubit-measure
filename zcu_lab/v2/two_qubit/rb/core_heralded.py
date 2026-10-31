"""Paired RB with a ground-ground herald before each sequence."""

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
from .analysis import RBAnalysis, RBOptions, analyze_rb
from .core import RBCfg, RBResult, acquire_counts


class RBHeraldedCfg(RBCfg):
    """Optional reset → herald → paired sequence → terminal, per host round.

    Fields and units match RBCfg. Both reads share the saved full mux cfg and
    circle radii. Its post_delay supplies the wait after herald. reset=None
    emits no reset; rounds stays a host axis rather than an acquisition average.
    """


@dataclass(frozen=True)
class RBHeraldedResult:
    """Raw float64 sums (depth,seed,arm,round,3,3,3,3), NaN when unacquired.

    depths, seeds, arms and rounds have the same meanings as RBResult.
    Trailing axes are herald F1/F2 then terminal F1/F2, each ordered g/e/other.
    All herald outcomes remain saved; counts are not normalized or filtered.
    """

    depths: NDArray[np.int64]
    seeds: NDArray[np.int64]
    arms: NDArray[np.int64]
    rounds: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class RBHeraldedAnalysis:
    """terminal is paired RB analysis conditioned on herald classified (g,g).

    herald_matrix is (depth,seed,arm,round,3,3), normalized over herald classes.
    pass_fraction is its (g,g) cell with the four scan axes, not state purity.
    Empty or unmeasured rounds give NaN probabilities.
    """

    terminal: RBAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(data: GroupedLoadData[RBHeraldedCfg]) -> RBHeraldedResult:
    """Restore all 81 class counts; old per-shot IQ files fail without sums."""
    sums = data.variable("sums")
    return RBHeraldedResult(
        depths=sums.axes[7].astype(np.int64),
        seeds=sums.axes[6].astype(np.int64),
        arms=sums.axes[5].astype(np.int64),
        rounds=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


class RBHeraldedExp(Experiment[RBHeraldedResult, RBHeraldedCfg]):
    """Paired RB with herald acceptance and the ordinary complete-round fit."""

    Options = RBOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[RBHeraldedResult, RBHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec.generated_arange("herald_f2", ""),
                        VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec("Round", "", field_name="rounds", dtype=np.int64),
                        VariableAxisSpec("Arm", "", field_name="arms", dtype=np.int64),
                        VariableAxisSpec("Seed", "", field_name="seeds", dtype=np.int64),
                        VariableAxisSpec(
                            "Clifford depth", "", field_name="depths", dtype=np.int64
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums", label="Herald and terminal class counts",
                        unit="", dtype=np.float64,
                    ),
                ),
            ),
            result_type=RBHeraldedResult,
            cfg_type=RBHeraldedCfg,
            tag="two_qubit/rb_heralded",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: RBHeraldedCfg, *, context: RunContext
    ) -> RBHeraldedResult:
        """Acquire optional reset → herald → paired sequence → terminal.

        Compile once per depth/seed/arm. Each host round acquires reps shots with
        rounds=1 and saves all 81 counts. Unfinished slots stay NaN. The caller
        checks context.scope.stop for acquisition errors; cfg remains unchanged.
        """
        return RBHeraldedResult(
            np.array(config.depths, dtype=np.int64),
            np.array(config.seeds, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.arange(config.rounds, dtype=np.int64),
            acquire_counts(config, context=context, heralded=True),
        )

    def analyze(
        self,
        source: RunRecord[RBHeraldedCfg, RBHeraldedResult],
        options: RBOptions,
        *,
        plots: Plots,
    ) -> RBHeraldedAnalysis:
        """Fit herald (g,g) terminal count ratios; show nine herald fractions.

        Complete-round selection and bootstrap options match ordinary RB.
        Zero accepted counts cannot enter paired fits. Herald probabilities use
        all saved counts and do not claim ground-state purity.
        """
        data = source.result
        shape = (len(data.depths), len(data.seeds), 2, len(data.rounds), 3, 3, 3, 3)
        if data.sums.shape != shape:
            raise ValueError("Expected heralded RB sums (depth,seed,2,round,3,3,3,3)")
        acquired = np.isfinite(data.sums).all(axis=(-4, -3, -2, -1))
        missing = np.isnan(data.sums).all(axis=(-4, -3, -2, -1))
        if not np.all(acquired | missing):
            raise ValueError("Heralded RB sums contains nonfinite counts or a partial round")
        terminal = analyze_rb(
            RunRecord[RBCfg, RBResult](
                cfg=source.cfg,
                result=RBResult(
                    data.depths, data.seeds, data.arms, data.rounds,
                    terminal_counts(data.sums, heralded=True),
                ),
            ),
            options,
            plots=plots,
        )
        matrix = herald_probabilities(data.sums)
        aggregate = herald_probabilities(np.nansum(data.sums, axis=(1, 3)))
        for arm, label in enumerate(("Reference", "CZ interleaved")):
            fig, ax = plots.subplots(f"herald_{arm}", figsize=(10, 6))
            for f1 in range(3):
                for f2 in range(3):
                    classes = ("g", "e", "other")
                    ax.plot(
                        data.depths, aggregate[:, arm, f1, f2], "o-",
                        label=f"{classes[f1]},{classes[f2]}",
                    )
            ax.set(
                xlabel="Random Clifford depth", ylabel="Herald probability",
                title=f"{label}: (g,g) is pass fraction, not purity",
            )
            ax.legend()
            fig.tight_layout()
        return RBHeraldedAnalysis(terminal, matrix, matrix[..., 0, 0])
