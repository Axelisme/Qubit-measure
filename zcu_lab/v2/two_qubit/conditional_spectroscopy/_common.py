"""Shared cfg fields and shot timeline for the two spectroscopy entries."""

from __future__ import annotations

from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick import QickConfig
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.program.v2 import Pulse, Reset, SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules import MuxHeraldResetCfg, MuxReadout, ResetCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .._drive import QubitDriveCfg
from .._sums import qubit_marginal_axis, validate_qubit_tones


class SpectroscopyScanCfg[R: ResetCfg | MuxHeraldResetCfg](ConfigBase):
    """Common scans and pulses; R defines which reset modules an entrance admits.

    Pulse frequencies are MHz, phases degrees, and delays µs. The two probe
    frequencies are replaced by the shared sweep. Contrast adds both Xπ pulses
    after the probe. DC is untouched when dc_device is None; otherwise the
    named source is set in dc_unit with no compensation or restore.
    """

    reset: R | None = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the Rabi-calibrated Qubit1 flip for nominal preparation and readout contrast.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the Rabi-calibrated Qubit2 flip for nominal preparation and readout contrast.",
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Two-qubit spectroscopy probe",
            description="Enter the two weak pulses applied together on the respective qubit drive lines at the shared RF sweep frequency.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Enter the full mux module used after probe and optional contrast, with GE Post circle radii for both qubits.",
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
    contrast: Annotated[
        bool,
        FieldMeta(
            label="Double Xπ contrast",
            description="Enable an Xπ on each qubit after the probe, immediately before readout.",
        ),
    ] = True
    dc_device: Annotated[
        DeviceName | None,
        FieldMeta(
            label="Coupler DC source",
            description="Select the connected coupler-bias source by name; leave blank to keep all sources unchanged.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="DC source unit",
            description="Select amperes, volts or the selected source's native unit for the bias sweep.",
        ),
    ] = "native"
    dc: Annotated[
        SweepCfg,
        FieldMeta(
            label="Coupler DC bias sweep",
            description="Enter the sweep in DC source unit (A, V or native); leave the source blank to ignore this sweep.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common probe RF frequency",
            description="Enter the RF sweep (MHz); each point replaces the frequency of both probe pulses.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter attempted shots per preparation and frequency per round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter repeated frequency scans at each DC bias point.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial wait",
            unit="µs",
            description="Enter the wait before the first shot of each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot before the next preparation.",
        ),
    ]

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


def build_spectroscopy_program[R: ResetCfg | MuxHeraldResetCfg](
    config: SpectroscopyScanCfg[R], *, soccfg: QickConfig, herald: bool
) -> TimelineProgramV2:
    """Build one bias point's preparation × frequency scan for the supplied board.

    Optional reset precedes both nominal flips. If herald is true, emit the mux
    with tag "herald" directly after the flips. Always tag the last read "terminal".
    Both reads use the same cfg and its wait. Invalid hardware settings raise
    during compile; this function does not set a bias or acquire.
    """
    builder = TimelineBuilder(
        soccfg,
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
    if herald:
        builder.emit(readout, tag="herald")
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
    return builder.build()


def excitation_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return terminal Qubit1/Qubit2 P(e) SEM; counts end in 3 × 3 (g/e/other).

    Other contributes to N. Fewer than two shots and unmeasured points give NaN.
    """
    errors = []
    for target in ("qubit1", "qubit2"):
        marginal = counts.sum(axis=qubit_marginal_axis(target))
        errors.append(np.sqrt(class_probability_covariance(marginal)[..., 1, 1]))
    return np.stack(errors, axis=-1)
