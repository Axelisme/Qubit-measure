"""Four-branch two-line spectroscopy with acquisition-side joint counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
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
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._drive import QubitDriveCfg
from .._readout import MuxResetCfg
from .._sums import (
    qubit_marginal_probabilities,
    validate_qubit_tones,
)


@dataclass(frozen=True)
class SpectroscopyResult:
    """Raw float64 sums (DC,2,2,frequency,3,3), terminal Qubit1/Qubit2 g/e/other.

    DC coordinates are device-native, or [0] when inactive. Frequencies are MHz.
    Preparation axes are fixed [0,1] (g/e), Qubit1 outer. Unmeasured points
    remain NaN; completed acquisition rounds survive cancellation.
    """

    dc_values: NDArray[np.float64]

    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class SpectroscopyAnalysis:
    """Count-derived spectra on (DC, prepared_qubit1, prepared_qubit2, frequency).

    excitation adds a final Qubit1/Qubit2 axis. acquired_shots is the per-point sum
    after any herald selection. complete marks DC/preparation rows with at
    least one completed round at every frequency. Zero totals give NaN
    excitation. Other stays in the denominator; probabilities are not
    corrected for state-preparation or readout error.
    """

    excitation: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    complete: NDArray[np.bool_]


class SpectroscopyCfg(ConfigBase):
    """Optional reset → B00/01/10/11 → dual probe → optional double Xπ → mux.

    Frequency is a common MHz hardware sweep; pulses carry explicit waveforms,
    gains and phases (degrees), with timing in us. contrast applies both Xπ
    pulses after the probe. Qubit1/Qubit2 require circle classifiers with GE Post radii.
    rounds accumulate counts at acquisition, not on a host round axis.
    dc_device=None leaves all sources untouched and uses the sentinel axis [0].
    Otherwise sweep the named source in dc_unit, without compensation or restore.
    """

    reset: Annotated[
        MuxResetCfg | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ", description="Calibrated Qubit1 preparation pulse."
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ", description="Calibrated Qubit2 preparation pulse."
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Two charge-line weak probe pulses; confirm gains and durations before Run.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full mux readout; Qubit1/Qubit2 need circle + radius from GE Post.",
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    contrast: Annotated[bool, FieldMeta(label="Double Xπ contrast")] = True
    dc_device: Annotated[DeviceName | None, FieldMeta(label="Coupler DC source")] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"], FieldMeta(label="DC source working unit")
    ] = "native"
    dc: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC source values",
            description="Inactive when source is None; no Qubit1/Qubit2 compensation.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="Common probe frequency scan on both charge lines.",
        ),
    ]
    reps: Annotated[int, FieldMeta(label="Reps")] = Field(gt=0)
    rounds: Annotated[int, FieldMeta(label="Rounds")] = Field(default=1, gt=0)
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = 1.0
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us", decimals=3)]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require calibrated qubit tones, matched probe lines and finite scans."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each probe must use its qubit's calibrated Xπ generator")
        if self.dc_device is not None and not self.dc_device.strip():
            raise ValueError("DC source must name a connected device or be None")
        return self


def summarize_spectra(counts: NDArray[np.float64]) -> SpectroscopyAnalysis:
    """Compute Qubit1/Qubit2 excitation, accepted totals and all-frequency coverage.

    counts ends in terminal (3,3), with g/e/other on each axis and frequency
    immediately before them. Counts accumulate over completed rounds. Whole
    missing points may be NaN. Invalid counts or partly missing points raise
    ValueError. Zero totals yield NaN excitation.
    """
    if counts.ndim < 3 or counts.shape[-2:] != (3, 3):
        raise ValueError("Expected a frequency axis and terminal class axes (3,3)")
    finite = np.all(np.isfinite(counts) & (counts >= 0), axis=(-2, -1))
    missing = np.all(np.isnan(counts), axis=(-2, -1))
    if not np.all(finite | missing):
        raise ValueError("Spectroscopy contains invalid or partially missing counts")
    acquired_shots = counts.sum(axis=(-2, -1))
    if not np.all(np.isfinite(acquired_shots) | missing):
        raise ValueError("Acquired shots must be nonnegative totals or NaN")
    excitation = np.stack(
        [
            qubit_marginal_probabilities(counts, target)[..., 1]
            for target in ("qubit1", "qubit2")
        ],
        axis=-1,
    )
    return SpectroscopyAnalysis(
        excitation,
        acquired_shots,
        np.all(finite, axis=-1),
    )


def restore_result(data: GroupedLoadData[SpectroscopyCfg]) -> SpectroscopyResult:
    """Restore summed counts; old per-shot IQ files lack sums and are rejected.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return SpectroscopyResult(
        dc_values=sums.axes[5], frequencies=sums.axes[2], sums=sums.z.astype(np.float64)
    )


def _plot_excitation(
    plots: Plots,
    dc: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    excitation: NDArray[np.float64],
) -> None:
    """Draw the Qubit1/Qubit2 excitation overlay per nominal preparation and DC point."""
    # Show the same preparation-resolved overlay as the live plot.
    for d, value in enumerate(dc):
        for qubit1, qubit2 in np.ndindex(2, 2):
            _, ax = plots.subplots(
                num=f"excitation_dc_{d}_prep_{qubit1}_{qubit2}", layout="constrained"
            )
            preparation = excitation[d, qubit1, qubit2]
            for values, label in zip(preparation.T, ("Qubit1", "Qubit2"), strict=True):
                ax.plot(frequencies, values, ".", label=label)
            ax.set(
                title=f"Nominal preparation ({qubit1},{qubit2}) · DC {value:g}",
                xlabel="Common probe frequency (MHz)",
                ylabel="Excited-state probability P_e",
            )
            ax.legend()


class SpectroscopyExp(Experiment[SpectroscopyResult, SpectroscopyCfg]):
    """Measure Qubit1/Qubit2 excitation versus common probe frequency in four branches."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[SpectroscopyResult, SpectroscopyCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec(
                            label="Probe frequency",
                            unit="Hz",
                            field_name="frequencies",
                            scale=1e6,
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit2",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit1",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="DC source value (inactive if dc_device=null)",
                            unit="a.u.",
                            field_name="dc_values",
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Joint class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=SpectroscopyResult,
            cfg_type=SpectroscopyCfg,
            cfg_schema_version="7.0",
            tag="two_qubit/conditional_spectroscopy",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: SpectroscopyCfg, *, context: RunContext
    ) -> SpectroscopyResult:
        """Acquire optional reset → nominal branch → probe → contrast → terminal.

        Return raw counts, not IQ or proportions. Unmeasured points remain NaN
        after cooperative cancellation. Acquisition exceptions propagate unchanged.
        The stop signal carries only stop requests and reason text.

        """
        # 1. Show Qubit1/Qubit2 excitation spectra by nominal preparation and optional DC setting.
        frequencies = sweep2array(config.frequency)
        dc_values = (
            np.array([0.0]) if config.dc_device is None else sweep2array(config.dc)
        )
        viewers = {}
        for d, dc in enumerate(dc_values):
            for qubit1 in range(2):
                for qubit2 in range(2):
                    fig, ax = context.plots.subplots(
                        num=f"measurement_dc_{d}_prep_{qubit1}_{qubit2}"
                    )
                    ax.set(
                        xlabel="Common probe frequency (MHz)",
                        ylabel="Excited-state probability P_e",
                        title=f"Nominal preparation ({qubit1},{qubit2}) · DC {dc:g}",
                    )
                    viewers[d, qubit1, qubit2] = live_line(ax, num_lines=2)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    for line, label in zip(
                        ax.get_lines(), ("Qubit1", "Qubit2"), strict=True
                    ):
                        line.set_label(label)
                    ax.legend()
                    fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            qubit1_pe = qubit_marginal_probabilities(sums, "qubit1")[..., 1]
            qubit2_pe = qubit_marginal_probabilities(sums, "qubit2")[..., 1]
            artists: list[Line2D] = []
            for (d, qubit1, qubit2), viewer in viewers.items():
                artists.extend(
                    viewer.set(
                        frequencies,
                        np.stack(
                            (qubit1_pe[d, qubit1, qubit2], qubit2_pe[d, qubit1, qubit2])
                        ),
                    )
                )
            return tuple(artists)

        # 2. Set DC on the host; scan preparations and common probe frequency in hardware before contrast/readout.
        buffer = SignalBuffer(
            (len(dc_values), 2, 2, len(frequencies), 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
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
                builder.add_loop("prepared_qubit1", 2)  # 0 = g, 1 = e (outer)
                builder.add_loop("prepared_qubit2", 2)  # 0 = g, 1 = e (inner)
                frequency = builder.add_sweep("frequency", config.frequency)
                x180_qubit1 = Pulse("prepare_qubit1", config.x180_qubit1)
                x180_qubit2 = Pulse("prepare_qubit2", config.x180_qubit2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset, terminal=readout))
                builder.fork("default", into=["qubit1", "qubit2"])
                with (
                    builder.branch(
                        by="prepared_qubit1", align_timeline=True, timeline="qubit1"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_qubit1)
                with (
                    builder.branch(
                        by="prepared_qubit2", align_timeline=True, timeline="qubit2"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_qubit2)
                builder.join("qubit1", "qubit2", into="default")
                builder.fork("default", into=["probe_qubit1_time", "probe_qubit2_time"])
                builder.emit(
                    Pulse("probe_qubit1", config.drive.qubit1, freq=frequency),
                    timeline="probe_qubit1_time",
                )
                builder.emit(
                    Pulse("probe_qubit2", config.drive.qubit2, freq=frequency),
                    timeline="probe_qubit2_time",
                )
                builder.join("probe_qubit1_time", "probe_qubit2_time", into="default")
                if config.contrast:
                    builder.emit(Pulse("contrast_qubit1", config.x180_qubit1))
                    builder.emit(Pulse("contrast_qubit2", config.x180_qubit2))
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(
                        prog,
                        [
                            Classify("terminal", config.qubit1_tone),
                            Classify("terminal", config.qubit2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        # 3. Keep raw joint counts on planned DC/frequency axes; unmeasured points remain NaN.
        return SpectroscopyResult(dc_values, frequencies, buffer.array)

    def analyze(
        self,
        source: RunRecord[SpectroscopyCfg, SpectroscopyResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot count-derived Qubit1/Qubit2 P_e with other retained in the denominator."""
        # 1. Use saved axes for preparation-resolved excitation spectra with other classes in the denominator.
        del options
        data = source.result
        cfg = source.cfg
        if cfg is None:
            raise ValueError("Spectroscopy analysis requires saved cfg")
        dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc)
        frequencies = sweep2array(cfg.frequency)
        if (
            not np.array_equal(data.dc_values, dc)
            or data.frequencies.shape != frequencies.shape
            or not np.allclose(
                data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
            )
            or data.sums.shape != (len(dc), 2, 2, len(frequencies), 3, 3)
        ):
            raise ValueError(
                "Spectroscopy requires saved axes and sums (DC,2,2,frequency,3,3)"
            )
        answer = summarize_spectra(data.sums)
        _plot_excitation(plots, dc, frequencies, answer.excitation)

        # 2. Compare preparation-resolved Qubit1/Qubit2 excitation and optional DC maps without fitting.
        for target, role in enumerate(("Qubit1", "Qubit2")):
            fig, _ = plots.subplots(
                num=f"spectra_{role.lower()}",
                nrows=2,
                ncols=2,
                figsize=(11, 7),
                squeeze=False,
                layout="constrained",
            )
            fig.suptitle(f"{role} P_e · double Xπ contrast {cfg.contrast}")
            for qubit1, qubit2 in np.ndindex(2, 2):
                ax = fig.axes[2 * qubit1 + qubit2]
                for d, value in enumerate(dc):
                    ax.plot(
                        frequencies,
                        answer.excitation[d, qubit1, qubit2, :, target],
                        marker=".",
                        label=f"DC {value:g} {cfg.dc_unit}"
                        if cfg.dc_device
                        else "DC inactive",
                    )
                ax.set(
                    title=f"Nominal preparation ({qubit1},{qubit2})",
                    xlabel="Common probe frequency (MHz)",
                    ylabel=f"{role} P_e",
                    ylim=(0, 1),
                )
                ax.legend(fontsize="small")
            if cfg.dc_device is not None:
                fig, _ = plots.subplots(
                    num=f"map_{role.lower()}",
                    nrows=2,
                    ncols=2,
                    figsize=(11, 7),
                    squeeze=False,
                    layout="constrained",
                )
                for qubit1, qubit2 in np.ndindex(2, 2):
                    ax = fig.axes[2 * qubit1 + qubit2]
                    image = ax.pcolormesh(
                        frequencies,
                        dc,
                        answer.excitation[:, qubit1, qubit2, :, target],
                        shading="nearest",
                        vmin=0,
                        vmax=1,
                    )
                    ax.grid(False)
                    fig.colorbar(image, ax=ax, label=f"{role} P_e")
                    ax.set(
                        title=f"Nominal preparation ({qubit1},{qubit2})",
                        xlabel="Common probe frequency (MHz)",
                        ylabel=f"{cfg.dc_device} ({cfg.dc_unit})",
                    )
        # 3. Show acquired-shot coverage separately from excitation, including partial counts.
        fig, _ = plots.subplots(
            num="shots",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle("Acquired shots per point · partial counts included")
        for qubit1, qubit2 in np.ndindex(2, 2):
            ax = fig.axes[2 * qubit1 + qubit2]
            for d, value in enumerate(dc):
                shots = answer.acquired_shots[d, qubit1, qubit2]
                label = f"DC {value:g}; complete={answer.complete[d, qubit1, qubit2]}"
                if np.any(shots == 0):
                    label += "; no accepted shots at some frequencies"
                ax.plot(frequencies, shots, label=label)
            ax.axhline(
                cfg.reps * cfg.rounds,
                color="gray",
                linestyle="--",
                label="Requested shots",
            )
            ax.set(
                title=f"Nominal preparation ({qubit1},{qubit2})",
                xlabel="Common probe frequency (MHz)",
                ylabel="Shots",
            )
            ax.legend(fontsize="small")
