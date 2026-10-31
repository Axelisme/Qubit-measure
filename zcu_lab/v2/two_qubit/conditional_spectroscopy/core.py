"""Four-branch two-line spectroscopy with acquisition-side joint counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick import QickConfig
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
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles
from .._sums import (
    marginal_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import SpectroscopyAnalysis, analyze_spectra

BRANCH_LABELS = ("00", "01", "10", "11")


class SpectroscopyCfg(ConfigBase):
    """Optional reset → B00/01/10/11 → dual probe → optional double Xπ → mux.

    Frequency is a common MHz host scan; pulses carry explicit waveforms,
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
    """Raw float64 sums (DC,4,frequency,3,3), ordered terminal F1/F2 g/e/other.

    DC coordinates are device-native, or [0] when inactive. Frequencies are MHz.
    branches=[0,1,2,3] labels nominal B00/01/10/11, F1 first. Unmeasured points
    remain NaN; completed acquisition rounds survive cancellation.
    """

    dc_values: NDArray[np.float64]
    branches: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


def build_program(
    cfg: SpectroscopyCfg,
    soccfg: QickConfig,
    branch: int,
    frequency: float,
    *,
    heralded: bool,
) -> TimelineProgramV2:
    """Compile fresh reset/herald, nominal preparation, parallel probe and readout.

    branch is 0..3. contrast adds F1 then F2 Xπ after the probe. Herald and
    terminal reuse the same mux object; its post_delay supplies the herald gap.
    No acquisition occurs and input pulses are unchanged.
    """
    if branch not in range(4):
        raise ValueError("branch must be in 0..3")
    validate_classifiers(cfg.readout, cfg.roles)
    builder = TimelineBuilder(
        soccfg,
        reps=cfg.reps,
        initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    readout = MuxReadout("readout", cfg.readout)
    if cfg.reset is not None:
        builder.emit(Reset("reset", cfg.reset))
    if heralded:
        builder.emit(readout, tag="herald")
    if branch & 2:
        builder.emit(Pulse("prepare_f1", cfg.x180_f1))
    if branch & 1:
        builder.emit(Pulse("prepare_f2", cfg.x180_f2))
    builder.emit(dual_line_drive(cfg.drive, frequency=frequency, name="probe"))
    if cfg.contrast:
        builder.emit(Pulse("contrast_f1", cfg.x180_f1))
        builder.emit(Pulse("contrast_f2", cfg.x180_f2))
    builder.emit(readout, tag="terminal")
    return builder.build()


def acquire_counts(
    config: SpectroscopyCfg, *, context: RunContext, heralded: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Share DC/branch/frequency acquisition; return DC, MHz and unchanged sums.

    Optional named DC is left at its last setting. Rounds accumulate in the
    acquisition backend. Live curves show F1/F2 P_e by DC and branch, selecting
    herald (g,g) only for display in the heralded version.
    """
    validate_classifiers(config.readout, config.roles)
    frequencies = sweep2array(config.frequency)
    dc_values = (
        np.array([0.0]) if config.dc_device is None else sweep2array(config.dc_values)
    )
    viewers = {
        (d, branch): live_line(
            context.plots,
            f"measurement_dc_{d}_branch_{branch}",
            "Common probe frequency (MHz)",
            "P_e (F1 line 1, F2 line 2)",
            title=f"B{branch:02b} · DC {dc:g}"
            + (" · herald (g,g)" if heralded else ""),
            num_lines=2,
        )
        for d, dc in enumerate(dc_values)
        for branch in range(4)
    }

    def update_view(sums: NDArray[np.float64]) -> None:
        counts = terminal_counts(sums, ("g", "g")) if heralded else sums
        f1 = marginal_probabilities(counts, "f1")[..., 1]
        f2 = marginal_probabilities(counts, "f2")[..., 1]
        for (d, branch), viewer in viewers.items():
            viewer.update(frequencies, np.stack((f1[d, branch], f2[d, branch])))

    classes = (3, 3, 3, 3) if heralded else (3, 3)
    buffer = SignalBuffer(
        (len(dc_values), 4, len(frequencies), *classes),
        dtype=np.float64,
        on_update=update_view,
    )
    with Schedule(config, buffer, scope=context.scope) as sched:
        for dc_value, dc_step in sched.scan("dc", dc_values.tolist()):
            if dc_step.cfg.dc_device is not None:
                set_flux(
                    context.devices[dc_step.cfg.dc_device],
                    dc_value,
                    unit=dc_step.cfg.dc_unit,
                    stop=context.scope.stop,
                )
            for branch, branch_step in dc_step.scan("branch", list(range(4))):
                for frequency, step in branch_step.scan(
                    "frequency", frequencies.tolist()
                ):
                    prog = build_program(
                        step.cfg, context.soccfg, branch, frequency, heralded=heralded
                    )
                    step.run_acquire_summed(
                        prog,
                        context.soc,
                        processor=shot_sums(
                            prog,
                            [
                                Classify(label, tone)
                                for label in (
                                    ("herald", "terminal")
                                    if heralded
                                    else ("terminal",)
                                )
                                for tone in (
                                    step.cfg.roles.f1_tone,
                                    step.cfg.roles.f2_tone,
                                )
                            ],
                        ),
                        rounds=step.cfg.rounds,
                        options=RunOptions(),
                    )
    return dc_values, frequencies, buffer.array


def restore_result(data: GroupedLoadData[SpectroscopyCfg]) -> SpectroscopyResult:
    """Restore summed counts; old per-shot IQ files lack sums and are rejected."""
    sums = data.variable("sums")
    return SpectroscopyResult(
        dc_values=sums.axes[4],
        branches=sums.axes[3].astype(np.int64),
        frequencies=sums.axes[2],
        sums=sums.z.astype(np.float64),
    )


class SpectroscopyExp(Experiment[SpectroscopyResult, SpectroscopyCfg]):
    """Measure F1/F2 excitation versus common probe frequency in four branches."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[SpectroscopyResult, SpectroscopyCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
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
                        label="Joint class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=SpectroscopyResult,
            cfg_type=SpectroscopyCfg,
            cfg_schema_version="4.0",
            tag="two_qubit/conditional_spectroscopy",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: SpectroscopyCfg, *, context: RunContext
    ) -> SpectroscopyResult:
        """Acquire optional reset → nominal branch → probe → contrast → terminal.

        Return raw counts, not IQ or proportions. Check the operation stop signal
        for acquisition errors; unmeasured points remain NaN after cancellation.
        """
        dc, frequencies, sums = acquire_counts(config, context=context, heralded=False)
        return SpectroscopyResult(dc, np.arange(4, dtype=np.int64), frequencies, sums)

    def analyze(
        self,
        source: RunRecord[SpectroscopyCfg, SpectroscopyResult],
        options: None,
        *,
        plots: Plots,
    ) -> SpectroscopyAnalysis:
        """Plot count-derived F1/F2 P_e with other retained in the denominator."""
        del options
        data = source.result
        return analyze_spectra(
            source.cfg,
            data,
            acquired_shots=data.sums.sum(axis=(-2, -1)),
            plots=plots,
        )
