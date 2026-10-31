"""Two-qubit spectroscopy with gg/ee herald-conditioned excitation spectra."""

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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Classify, Pulse, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._sums import (
    herald_probabilities,
    marginal_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import SpectroscopyAnalysis, summarize_spectra
from .core import SpectroscopyCfg


class SpectroscopyHeraldedCfg(SpectroscopyCfg):
    """Optional reset → herald → nominal branch → probe → contrast → terminal.

    Both reads use the same mux settings and circle radii. Its post_delay is the
    wait after herald. Other fields and units match SpectroscopyCfg.
    """


@dataclass(frozen=True)
class SpectroscopyHeraldedResult:
    """Raw float64 sums (DC,2,2,frequency,3,3,3,3), never proportions.

    Class axes are herald F1/F2 then terminal F1/F2, each g/e/other.
    prepared_f1/prepared_f2 are [0,1] (g/e), F1 outer. DC is [0] when inactive;
    frequency is MHz. Missing points remain NaN.
    """

    dc_values: NDArray[np.float64]
    prepared_f1: NDArray[np.int64]
    prepared_f2: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class SpectroscopyHeraldedAnalysis:
    """gg/ee contain P_e, all-shot totals and completeness on nominal axes.

    Plotting aligns the two paths by actual preparation before the probe:
    nominal (f1,f2) for gg, (1-f1,1-f2) for ee. Neither path is fitted.
    herald_matrix is (DC,2,2,frequency,3,3); pass_fraction is its gg cell,
    not ground-state purity. Zero selected counts give NaN P_e.
    """

    gg: SpectroscopyAnalysis
    ee: SpectroscopyAnalysis
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[SpectroscopyHeraldedCfg],
) -> SpectroscopyHeraldedResult:
    """Restore all 81 counts; old per-shot IQ files lack sums and are rejected."""
    sums = data.variable("sums")
    return SpectroscopyHeraldedResult(
        dc_values=sums.axes[7],
        prepared_f1=sums.axes[6].astype(np.int64),
        prepared_f2=sums.axes[5].astype(np.int64),
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
                        "Prepared F2", "", field_name="prepared_f2", dtype=np.int64
                    ),
                    VariableAxisSpec(
                        "Prepared F1", "", field_name="prepared_f1", dtype=np.int64
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
        cfg_schema_version="2.0",
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
        validate_classifiers(config.readout, config.roles)
        frequencies = sweep2array(config.frequency)
        dc_values = (
            np.array([0.0])
            if config.dc_device is None
            else sweep2array(config.dc_values)
        )
        viewers = {
            (d, f1, f2): live_line(
                context.plots,
                f"measurement_dc_{d}_actual_{f1}_{f2}",
                "Common probe frequency (MHz)",
                "P_e (gg F1/F2, ee F1/F2)",
                title=f"Actual initial state ({f1},{f2}) before probe · DC {dc:g}",
                num_lines=4,
            )
            for d, dc in enumerate(dc_values)
            for f1 in range(2)
            for f2 in range(2)
        }

        def update_view(sums: NDArray[np.float64]) -> None:
            gg = terminal_counts(sums, ("g", "g"))
            ee = terminal_counts(sums, ("e", "e"))
            gg_f1 = marginal_probabilities(gg, "f1")[..., 1]
            gg_f2 = marginal_probabilities(gg, "f2")[..., 1]
            ee_f1 = marginal_probabilities(ee, "f1")[..., 1]
            ee_f2 = marginal_probabilities(ee, "f2")[..., 1]
            for (d, f1, f2), viewer in viewers.items():
                viewer.update(
                    frequencies,
                    np.stack(
                        (
                            gg_f1[d, f1, f2],
                            gg_f2[d, f1, f2],
                            ee_f1[d, 1 - f1, 1 - f2],
                            ee_f2[d, 1 - f1, 1 - f2],
                        )
                    ),
                )

        buffer = SignalBuffer(
            (len(dc_values), 2, 2, len(frequencies), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(config, buffer, scope=context.scope) as sched:
            # DC stays on the host because each point sets an external instrument.
            for dc_value, step in sched.scan("dc", dc_values.tolist()):
                if config.dc_device is not None:
                    set_flux(
                        context.devices[config.dc_device],
                        dc_value,
                        unit=config.dc_unit,
                        stop=context.scope.stop,
                    )
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("prepared_f1", 2)  # 0 = g, 1 = e (outer)
                builder.add_loop("prepared_f2", 2)  # 0 = g, 1 = e (inner)
                frequency = builder.add_sweep("frequency", config.frequency)
                x180_f1 = Pulse("prepare_f1", config.x180_f1)
                x180_f2 = Pulse("prepare_f2", config.x180_f2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")
                builder.fork("default", into=["f1", "f2"])
                with (
                    builder.timeline("f1"),
                    builder.branch(by="prepared_f1", align_timeline=True) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_f1)
                with (
                    builder.timeline("f2"),
                    builder.branch(by="prepared_f2", align_timeline=True) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_f2)
                builder.join("f1", "f2", into="default")
                builder.emit(
                    Join(
                        Pulse("probe_f1", config.drive.f1, freq=frequency),
                        Pulse("probe_f2", config.drive.f2, freq=frequency),
                    )
                )
                if config.contrast:
                    builder.emit(Pulse("contrast_f1", config.x180_f1))
                    builder.emit(Pulse("contrast_f2", config.x180_f2))
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(
                        prog,
                        [
                            Classify("herald", config.roles.f1_tone),
                            Classify("herald", config.roles.f2_tone),
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        return SpectroscopyHeraldedResult(
            dc_values,
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            frequencies,
            buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[SpectroscopyHeraldedCfg, SpectroscopyHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> SpectroscopyHeraldedAnalysis:
        """Overlay gg/ee spectra by actual initial state and show herald/shot totals.

        All completed rounds enter these diagnostics, including partial scans.
        Other counts remain in denominators. No fits or writeback are produced.
        """
        del options
        data, cfg = source.result, source.cfg
        if cfg is None:
            raise ValueError("Heralded spectroscopy analysis requires saved cfg")
        dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
        frequencies = sweep2array(cfg.frequency)
        if (
            not np.array_equal(data.dc_values, dc)
            or not np.array_equal(data.prepared_f1, np.arange(2))
            or not np.array_equal(data.prepared_f2, np.arange(2))
            or data.frequencies.shape != frequencies.shape
            or not np.allclose(
                data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
            )
            or data.sums.shape != (len(dc), 2, 2, len(frequencies), 3, 3, 3, 3)
        ):
            raise ValueError(
                "Heralded spectroscopy requires saved axes and sums (DC,2,2,frequency,3,3,3,3)"
            )
        finite = np.all(
            np.isfinite(data.sums) & (data.sums >= 0), axis=(-4, -3, -2, -1)
        )
        missing = np.all(np.isnan(data.sums), axis=(-4, -3, -2, -1))
        if not np.all(finite | missing):
            raise ValueError(
                "Heralded spectroscopy contains invalid or partially missing counts"
            )
        acquired = data.sums.sum(axis=(-4, -3, -2, -1))
        gg_counts = terminal_counts(data.sums, ("g", "g"))
        ee_counts = terminal_counts(data.sums, ("e", "e"))
        gg = summarize_spectra(gg_counts, acquired, cfg.reps * cfg.rounds)
        ee = summarize_spectra(ee_counts, acquired, cfg.reps * cfg.rounds)
        for d, value in enumerate(dc):
            for target, role in enumerate(("F1", "F2")):
                fig, _ = plots.subplots(
                    f"spectra_dc_{d}_{role.lower()}",
                    nrows=2,
                    ncols=2,
                    figsize=(11, 7),
                    squeeze=False,
                    layout="constrained",
                )
                fig.suptitle(
                    f"{role} P_e · DC {value:g} {cfg.dc_unit if cfg.dc_device else '(inactive)'}"
                    f" · double Xπ contrast {cfg.contrast}"
                )
                for f1, f2 in np.ndindex(2, 2):
                    ax = fig.axes[2 * f1 + f2]
                    ax.plot(
                        frequencies,
                        gg.excitation[d, f1, f2, :, target],
                        label=f"herald gg, nominal ({f1},{f2})",
                    )
                    ax.plot(
                        frequencies,
                        ee.excitation[d, 1 - f1, 1 - f2, :, target],
                        linestyle="--",
                        label=f"herald ee, nominal ({1 - f1},{1 - f2})",
                    )
                    ax.set(
                        title=f"Actual initial state ({f1},{f2}) before probe",
                        xlabel="Common probe frequency (MHz)",
                        ylabel=f"{role} P_e",
                        ylim=(0, 1),
                    )
                    ax.legend(fontsize="small")
        fig, _ = plots.subplots(
            "shots",
            nrows=len(dc),
            ncols=4,
            figsize=(16, 3 * len(dc)),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle("Shots per point · nominal preparation · partial counts included")
        gg_shots = gg_counts.sum(axis=(-2, -1))
        ee_shots = ee_counts.sum(axis=(-2, -1))
        for d, value in enumerate(dc):
            for f1, f2 in np.ndindex(2, 2):
                ax = fig.axes[4 * d + 2 * f1 + f2]
                ax.plot(frequencies, acquired[d, f1, f2], label="All 81 classes")
                ax.plot(frequencies, gg_shots[d, f1, f2], label="Selected gg")
                ax.plot(frequencies, ee_shots[d, f1, f2], label="Selected ee")
                ax.axhline(
                    cfg.reps * cfg.rounds,
                    color="gray",
                    linestyle="--",
                    label="Requested",
                )
                ax.set(
                    title=f"DC {value:g} · nominal ({f1},{f2}) · complete={gg.complete[d, f1, f2]}",
                    xlabel="Common probe frequency (MHz)",
                    ylabel="Shots",
                )
                ax.legend(fontsize="small")
        herald = herald_probabilities(data.sums)
        fig, _ = plots.subplots(
            "herald",
            nrows=len(dc),
            ncols=4,
            figsize=(16, 3 * len(dc)),
            squeeze=False,
            layout="constrained",
        )
        outcomes = [
            f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
        ]
        fig.suptitle(
            "Herald distribution; gg is pass fraction, not ground-state purity"
        )
        for d, value in enumerate(dc):
            for f1, f2 in np.ndindex(2, 2):
                ax = fig.axes[4 * d + 2 * f1 + f2]
                image = ax.pcolormesh(
                    frequencies,
                    np.arange(9),
                    herald[d, f1, f2].reshape(-1, 9).T,
                    shading="nearest",
                    vmin=0,
                    vmax=1,
                )
                ax.set(
                    title=f"DC {value:g} · nominal ({f1},{f2})",
                    xlabel="Common probe frequency (MHz)",
                    ylabel="Herald class (F1, F2)",
                    yticks=range(9),
                    yticklabels=outcomes,
                )
                fig.colorbar(image, ax=ax, label="Herald probability")
        return SpectroscopyHeraldedAnalysis(gg, ee, herald, herald[..., 0, 0])
