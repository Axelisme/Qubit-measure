"""Four-branch spectroscopy conditioned on a classified ground-ground herald."""

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
from .analysis import SpectroscopyAnalysis, analyze_spectra
from .core import SpectroscopyCfg, SpectroscopyResult, acquire_counts


class SpectroscopyHeraldedCfg(SpectroscopyCfg):
    """Optional reset → herald → nominal branch → probe → contrast → terminal.

    Both reads use the same mux settings and circle radii. Its post_delay is the
    wait after herald. Other fields and units match SpectroscopyCfg.
    """


@dataclass(frozen=True)
class SpectroscopyHeraldedResult:
    """Raw float64 sums (DC,4,frequency,3,3,3,3), never proportions.

    Class axes are herald F1/F2 then terminal F1/F2, each g/e/other.
    DC is [0] when inactive; frequency is MHz. Missing points remain NaN.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class SpectroscopyHeraldedAnalysis:
    """terminal is shared P_e analysis of herald (g,g) selected counts.

    Its acquired_shots totals and complete flags describe all acquired shots,
    not just selected shots.
    herald_matrix is (DC,4,frequency,3,3); pass_fraction is its (g,g) cell,
    not ground-state purity. Zero acceptance gives NaN terminal P_e.
    """

    terminal: SpectroscopyAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[SpectroscopyHeraldedCfg],
) -> SpectroscopyHeraldedResult:
    """Restore all 81 counts; old per-shot IQ files lack sums and are rejected."""
    sums = data.variable("sums")
    return SpectroscopyHeraldedResult(
        dc_values=sums.axes[6],
        branches=sums.axes[5].astype(np.int64),
        frequencies=sums.axes[4],
        sums=sums.z.astype(np.float64),
    )


class SpectroscopyHeraldedExp(
    Experiment[SpectroscopyHeraldedResult, SpectroscopyHeraldedCfg]
):
    """Measure conditional F1/F2 excitation spectra and herald acceptance."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[SpectroscopyHeraldedResult, SpectroscopyHeraldedCfg]
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
                        "Probe frequency", "Hz", field_name="frequencies", scale=1e6
                    ),
                    VariableAxisSpec(
                        "Nominal branch", "", field_name="branches", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        "DC source value (inactive if dc_device=null)",
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
        result_type=SpectroscopyHeraldedResult,
        cfg_type=SpectroscopyHeraldedCfg,
        tag="two_qubit/conditional_spectroscopy_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: SpectroscopyHeraldedCfg, *, context: RunContext
    ) -> SpectroscopyHeraldedResult:
        """Accumulate all herald/terminal counts across acquisition rounds.

        Optional DC settings are not restored. Completed rounds survive stop;
        check the operation stop signal for transport errors.
        """
        dc, frequencies, sums = acquire_counts(config, context=context, heralded=True)
        return SpectroscopyHeraldedResult(
            dc, np.arange(4, dtype=np.int64), frequencies, sums
        )

    def analyze(
        self,
        source: RunRecord[SpectroscopyHeraldedCfg, SpectroscopyHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> SpectroscopyHeraldedAnalysis:
        """Plot shared selected P_e spectra plus nine herald fractions per point."""
        del options
        data, cfg = source.result, source.cfg
        expected = (len(data.dc_values), 4, len(data.frequencies), 3, 3, 3, 3)
        if data.sums.shape != expected:
            raise ValueError(
                "Heralded spectroscopy requires sums (DC,4,frequency,3,3,3,3)"
            )
        finite = np.all(
            np.isfinite(data.sums) & (data.sums >= 0), axis=(-4, -3, -2, -1)
        )
        missing = np.all(np.isnan(data.sums), axis=(-4, -3, -2, -1))
        if not np.all(finite | missing):
            raise ValueError(
                "Heralded spectroscopy contains invalid counts or a partially missing point"
            )
        selected = SpectroscopyResult(
            data.dc_values,
            data.branches,
            data.frequencies,
            terminal_counts(data.sums, ("g", "g")),
        )
        terminal = analyze_spectra(
            cfg,
            selected,
            acquired_shots=data.sums.sum(axis=(-4, -3, -2, -1)),
            plots=plots,
        )
        herald = herald_probabilities(data.sums)
        fig, _ = plots.subplots(
            "herald",
            nrows=len(data.dc_values),
            ncols=4,
            figsize=(16, 3 * len(data.dc_values)),
            squeeze=False,
            layout="constrained",
        )
        outcomes = [
            f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
        ]
        fig.suptitle(
            "Herald distribution; (g,g) is pass fraction, not ground-state purity"
        )
        for d, dc in enumerate(data.dc_values):
            for branch in range(4):
                ax = fig.axes[d * 4 + branch]
                image = ax.pcolormesh(
                    data.frequencies,
                    np.arange(9),
                    herald[d, branch].reshape(-1, 9).T,
                    shading="nearest",
                    vmin=0,
                    vmax=1,
                )
                ax.set(
                    title=f"B{branch:02b} · DC {dc:g}",
                    xlabel="Common probe frequency (MHz)",
                    ylabel="Herald class (F1, F2)",
                    yticks=range(9),
                    yticklabels=outcomes,
                )
                fig.colorbar(image, ax=ax, label="Herald probability")
        return SpectroscopyHeraldedAnalysis(terminal, herald, herald[..., 0, 0])
