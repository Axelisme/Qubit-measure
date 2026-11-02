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

from .._drive import DualLineCfg
from .._readout import JointRoles
from .._sums import (
    marginal_probabilities,
    validate_classifiers,
)
from .analysis import summarize_spectra


class SpectroscopyCfg(ConfigBase):
    """Optional reset → B00/01/10/11 → dual probe → optional double Xπ → mux.

    Frequency is a common MHz hardware sweep; pulses carry explicit waveforms,
    gains and phases (degrees), with timing in us. contrast applies both Xπ
    pulses after the probe. F1/F2 require circle classifiers with GE Post radii.
    rounds accumulate counts at acquisition, not on a host round axis.
    dc_device=None leaves all sources untouched and uses the sentinel axis [0].
    Otherwise sweep the named source in dc_unit, without compensation or restore.
    """

    reps: Annotated[int, FieldMeta(label="Reps")] = Field(default=1, gt=0)
    rounds: Annotated[int, FieldMeta(label="Rounds")] = Field(default=1, gt=0)
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = 1.0
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset", description="Optional reset at the start of every shot."
        ),
    ] = None
    x180_f1: Annotated[
        Pulse.Cfg,
        FieldMeta(label="F1 Xπ", description="Calibrated F1 preparation pulse."),
    ]
    x180_f2: Annotated[
        Pulse.Cfg,
        FieldMeta(label="F2 Xπ", description="Calibrated F2 preparation pulse."),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Full mux readout; F1/F2 need circle + radius from GE Post.",
        ),
    ]
    roles: JointRoles
    drive: Annotated[
        DualLineCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Two charge-line weak probe pulses; confirm gains and durations before Run.",
        ),
    ]
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="Common probe frequency scan on both charge lines.",
        ),
    ]
    contrast: Annotated[bool, FieldMeta(label="Double Xπ contrast")] = True
    dc_device: Annotated[DeviceName | None, FieldMeta(label="Coupler DC source")] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"], FieldMeta(label="DC source working unit")
    ] = "native"
    dc_values: Annotated[
        SweepCfg,
        FieldMeta(
            label="DC source values",
            description="Inactive when source is None; no F1/F2 compensation.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require calibrated role tones, matched probe lines and finite scans."""
        validate_classifiers(self.readout, self.roles)
        if self.drive.f1.ch != self.x180_f1.ch or self.drive.f2.ch != self.x180_f2.ch:
            raise ValueError("Each probe must use its role's calibrated Xπ generator")
        if not np.all(np.isfinite(sweep2array(self.frequency))):
            raise ValueError("Probe frequencies must be finite")
        if self.dc_device is not None:
            if not self.dc_device.strip():
                raise ValueError("DC source must name a connected device or be None")
            if not np.all(np.isfinite(sweep2array(self.dc_values))):
                raise ValueError("DC source coordinates must be finite")
        return self


@dataclass(frozen=True)
class SpectroscopyResult:
    """Raw float64 sums (DC,2,2,frequency,3,3), terminal F1/F2 g/e/other.

    DC coordinates are device-native, or [0] when inactive. Frequencies are MHz.
    Preparation axes are fixed [0,1] (g/e), F1 outer. Unmeasured points
    remain NaN; completed acquisition rounds survive cancellation.
    """

    dc_values: NDArray[np.float64]

    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(data: GroupedLoadData[SpectroscopyCfg]) -> SpectroscopyResult:
    """Restore summed counts; old per-shot IQ files lack sums and are rejected."""
    sums = data.variable("sums")
    return SpectroscopyResult(
        dc_values=sums.axes[5], frequencies=sums.axes[2], sums=sums.z.astype(np.float64)
    )


class SpectroscopyExp(Experiment[SpectroscopyResult, SpectroscopyCfg]):
    """Measure F1/F2 excitation versus common probe frequency in four branches."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[SpectroscopyResult, SpectroscopyCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                        VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                        VariableAxisSpec(
                            label="Probe frequency",
                            unit="Hz",
                            field_name="frequencies",
                            scale=1e6,
                        ),
                        VariableAxisSpec(
                            label="Prepared F2", unit="", values=(0, 1), dtype=np.int64
                        ),
                        VariableAxisSpec(
                            label="Prepared F1", unit="", values=(0, 1), dtype=np.int64
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
            cfg_schema_version="5.0",
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
        validate_classifiers(config.readout, config.roles)
        frequencies = sweep2array(config.frequency)
        dc_values = (
            np.array([0.0])
            if config.dc_device is None
            else sweep2array(config.dc_values)
        )
        viewers = {}
        for d, dc in enumerate(dc_values):
            for f1 in range(2):
                for f2 in range(2):
                    fig, ax = context.plots.subplots(
                        num=f"measurement_dc_{d}_prep_{f1}_{f2}"
                    )
                    ax.set(
                        xlabel="Common probe frequency (MHz)",
                        ylabel="P_e (F1 line 1, F2 line 2)",
                        title=f"Nominal preparation ({f1},{f2}) · DC {dc:g}",
                    )
                    viewers[d, f1, f2] = live_line(ax, num_lines=2)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.grid(False)
                    fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            f1_pe = marginal_probabilities(sums, "f1")[..., 1]
            f2_pe = marginal_probabilities(sums, "f2")[..., 1]
            artists: list[Line2D] = []
            for (d, f1, f2), viewer in viewers.items():
                artists.extend(
                    viewer.set(
                        frequencies, np.stack((f1_pe[d, f1, f2], f2_pe[d, f1, f2]))
                    )
                )
            return tuple(artists)

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
                builder.add_loop("prepared_f1", 2)  # 0 = g, 1 = e (outer)
                builder.add_loop("prepared_f2", 2)  # 0 = g, 1 = e (inner)
                frequency = builder.add_sweep("frequency", config.frequency)
                x180_f1 = Pulse("prepare_f1", config.x180_f1)
                x180_f2 = Pulse("prepare_f2", config.x180_f2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.fork("default", into=["f1", "f2"])
                with (
                    builder.branch(
                        by="prepared_f1", align_timeline=True, timeline="f1"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_f1)
                with (
                    builder.branch(
                        by="prepared_f2", align_timeline=True, timeline="f2"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_f2)
                builder.join("f1", "f2", into="default")
                builder.fork("default", into=["probe_f1_time", "probe_f2_time"])
                builder.emit(
                    Pulse("probe_f1", config.drive.f1, freq=frequency),
                    timeline="probe_f1_time",
                )
                builder.emit(
                    Pulse("probe_f2", config.drive.f2, freq=frequency),
                    timeline="probe_f2_time",
                )
                builder.join("probe_f1_time", "probe_f2_time", into="default")
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
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        return SpectroscopyResult(dc_values, frequencies, buffer.array)

    def analyze(
        self,
        source: RunRecord[SpectroscopyCfg, SpectroscopyResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot count-derived F1/F2 P_e with other retained in the denominator."""
        del options
        data = source.result
        cfg = source.cfg
        if cfg is None:
            raise ValueError("Spectroscopy analysis requires saved cfg")
        dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
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
        answer = summarize_spectra(
            data.sums, data.sums.sum(axis=(-2, -1)), cfg.reps * cfg.rounds
        )
        for target, role in enumerate(("F1", "F2")):
            fig, _ = plots.subplots(
                num=f"spectra_{role.lower()}",
                nrows=2,
                ncols=2,
                figsize=(11, 7),
                squeeze=False,
                layout="constrained",
            )
            fig.suptitle(f"{role} P_e · double Xπ contrast {cfg.contrast}")
            for f1, f2 in np.ndindex(2, 2):
                ax = fig.axes[2 * f1 + f2]
                for d, value in enumerate(dc):
                    ax.plot(
                        frequencies,
                        answer.excitation[d, f1, f2, :, target],
                        label=f"DC {value:g} {cfg.dc_unit}"
                        if cfg.dc_device
                        else "DC inactive",
                    )
                ax.set(
                    title=f"Nominal preparation ({f1},{f2})",
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
                for f1, f2 in np.ndindex(2, 2):
                    ax = fig.axes[2 * f1 + f2]
                    image = ax.pcolormesh(
                        frequencies,
                        dc,
                        answer.excitation[:, f1, f2, :, target],
                        shading="nearest",
                        vmin=0,
                        vmax=1,
                    )
                    fig.colorbar(image, ax=ax, label=f"{role} P_e")
                    ax.set(
                        title=f"Nominal preparation ({f1},{f2})",
                        xlabel="Common probe frequency (MHz)",
                        ylabel=f"{cfg.dc_device} ({cfg.dc_unit})",
                    )
        fig, _ = plots.subplots(
            num="shots",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle("Acquired shots per point · partial counts included")
        for f1, f2 in np.ndindex(2, 2):
            ax = fig.axes[2 * f1 + f2]
            for d, value in enumerate(dc):
                ax.plot(
                    frequencies,
                    answer.acquired_shots[d, f1, f2],
                    label=f"DC {value:g}; complete={answer.complete[d, f1, f2]}",
                )
            ax.axhline(
                cfg.reps * cfg.rounds,
                color="gray",
                linestyle="--",
                label="Requested shots",
            )
            ax.set(
                title=f"Nominal preparation ({f1},{f2})",
                xlabel="Common probe frequency (MHz)",
                ylabel="Shots",
            )
            ax.legend(fontsize="small")
