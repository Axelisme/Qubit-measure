"""Herald-conditioned idle ZZ from acquisition-side class sums."""

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
from ._analysis import IdleZZAnalysis
from .core import (
    IdleZZCfg,
    IdleZZResult,
    acquire_counts,
    analyze_counts,
    complete_points,
)


class IdleZZHeraldedCfg(IdleZZCfg):
    """IdleZZCfg with a full mux herald before spectator preparation.

    reset=None emits no reset. Both reads use readout and its circle radii;
    readout.post_delay supplies the wait after herald. target is f1 or f2.
    """


@dataclass(frozen=True)
class IdleZZHeraldedResult:
    """Axes match IdleZZResult; sums is float64 (D,2,T,3,3,3,3).

    Class order is herald F1, herald F2, terminal F1, terminal F2, each g/e/other.
    All herald classes are saved unnormalized. Unmeasured points remain NaN.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class IdleZZHeraldedAnalysis:
    """terminal fits herald (g,g) sigma with the ordinary idle-ZZ policy.

    herald_matrix is (D,2,T,3,3), all nine herald probabilities. pass_fraction
    is its (g,g) cell (D,2,T), not ground-state purity. Empty points give NaN.
    """

    terminal: IdleZZAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(data: GroupedLoadData[IdleZZHeraldedCfg]) -> IdleZZHeraldedResult:
    """Restore 81-class sums; old raw-IQ files missing sums raise ValueError."""
    sums = data.variable("sums")
    return IdleZZHeraldedResult(
        sums.axes[6].astype(np.float64),
        sums.axes[5].astype(np.int64),
        sums.axes[4].astype(np.float64),
        sums.z.astype(np.float64),
    )


class IdleZZHeraldedExp(Experiment[IdleZZHeraldedResult, IdleZZHeraldedCfg]):
    """Estimate conditional fringe differences after a classified (g,g) herald."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZHeraldedResult, IdleZZHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec.generated_arange("herald_f2", ""),
                        VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec("Delay", "s", field_name="delays", scale=1e-6),
                        VariableAxisSpec(
                            "Spectator branch",
                            "",
                            field_name="branches",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            "DC source value (inactive if disabled)",
                            "a.u.",
                            field_name="dc_values",
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
            result_type=IdleZZHeraldedResult,
            cfg_type=IdleZZHeraldedCfg,
            tag="two_qubit/idle_zz_heralded",
            result_builder=restore_result,
        )
    )

    def run(
        self,
        config: IdleZZHeraldedCfg,
        *,
        context: RunContext,
    ) -> IdleZZHeraldedResult:
        """Acquire reset if set → herald → spectator I/Xπ → Ramsey → terminal.

        The same mux object emits both reads. Return all 81 class counts, with
        rounds accumulated on the acquisition side. No writeback occurs.
        """
        return IdleZZHeraldedResult(
            *acquire_counts(config, context=context, heralded=True)
        )

    def analyze(
        self,
        source: RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> IdleZZHeraldedAnalysis:
        """Fit selected target sigma and plot nine herald fractions per DC/branch.

        Completeness uses all saved classes, not the acceptance count. A complete
        acquisition with no accepted shots leaves ζ unavailable with a reason.
        Pass fraction is classified acceptance, not ground-state purity.
        """
        del options
        data = source.result
        complete = complete_points(source.cfg, data.sums, heralded=True)
        terminal = analyze_counts(
            source.cfg,
            IdleZZResult(
                data.dc_values,
                data.branches,
                data.delays,
                terminal_counts(data.sums, ("g", "g")),
            ),
            complete=complete,
            plots=plots,
        )
        matrix = herald_probabilities(data.sums)
        labels = [
            f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
        ]
        for dc_index, dc_value in enumerate(data.dc_values):
            for branch in range(2):
                fig, ax = plots.subplots(f"herald_{dc_index}_{branch}", figsize=(9, 5))
                for outcome, label in enumerate(labels):
                    ax.plot(
                        data.delays,
                        matrix[dc_index, branch].reshape(-1, 9)[:, outcome],
                        label=label,
                    )
                ax.set(
                    xlabel="Delay (us)",
                    ylabel="Herald probability",
                    title=f"DC={dc_value:g}; spectator {'I' if branch == 0 else 'Xπ'}",
                    ylim=(0, 1),
                )
                ax.legend()
                fig.suptitle("(g,g) is pass fraction, not ground-state purity")
                fig.tight_layout()
        return IdleZZHeraldedAnalysis(terminal, matrix, matrix[..., 0, 0])
