"""Local Z from target-resolved herald and terminal X/Y counts."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import QubitDriveCfg
from zcu_lab.v2.support.paired_ramsey import (
    count_phase,
    count_quadratures,
    phase_variance,
)
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.phase_line import fit_phase_line
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Reset
from zcu_tools.program.v2.modules import MuxReadout

from ..counts import terminal_counts

_TARGETS = ("qubit1", "qubit2")


@dataclass(frozen=True)
class LocalZHeraldedResult:
    """Int64 CZ gate_counts and float64 sums (2, 2, N, 3, 3, 3, 3).

    Axes are target, X/Y, train length, herald Qubit1/Qubit2, and terminal
    Qubit1/Qubit2. Class order is g/e/other; unmeasured points stay NaN.
    """

    gate_counts: NDArray[np.int64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class LocalZJointFit:
    """Shared slope and its 1σ error (deg/CZ), plus the negative-slope correction (deg).

    An unavailable fit has all numeric fields None and a reason. fit_quality
    describes the residuals and uncertainty of the joint model, not an acceptance gate.
    """

    slope_deg: float | None = None
    correction_deg: float | None = None
    slope_error_deg: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZGGFit:
    """Ordinary phase-line diagnostics on gg only, never used for correction.

    intercept_deg and correction_deg are in degrees; slope_deg and its 1σ
    slope_error_deg are in degrees/CZ. If fit is unavailable, reason explains
    the failure and fit_quality and all numeric fields are None.
    """

    intercept_deg: float | None
    slope_deg: float | None
    correction_deg: float | None
    slope_error_deg: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class LocalZSpectatorComparison:
    """Spectator e minus spectator g slope, in degrees/CZ and wrapped to ±180°.

    delta_deg_per_gate is None with a reason if either required slope is unavailable.
    """

    delta_deg_per_gate: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZHeraldedAnalysis:
    """Two-target summaries ordered Qubit1 then Qubit2 in each tuple.

    fits are the spectator-g calibration; gg_only is the ordinary comparison;
    spectator_comparisons report the change from an excited spectator.
    """

    fits: tuple[LocalZJointFit, LocalZJointFit]
    gg_only: tuple[LocalZGGFit, LocalZGGFit]
    spectator_comparisons: tuple[LocalZSpectatorComparison, LocalZSpectatorComparison]


class LocalZHeraldedCfg(ConfigBase):
    """Two-target CZ trains after an explicit mux herald and optional preparation reset.

    Gate counts are consecutive nonnegative integers. Reps are attempted shots
    per projection and round. Both ordered tones require circle classifiers with
    radii from GE Post. Mux post-delay applies after herald and terminal.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Optional preparation reset before the herald readout.",
        ),
    ] = None
    x90_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ/2", description="Prepare and project Qubit1 coherence."
        ),
    ]
    x90_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ/2", description="Prepare and project Qubit2 coherence."
        ),
    ]
    drive: Annotated[
        QubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Use both calibrated drive lines for every CZ gate.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Use the full calibrated mux readout for both reads. "
            + POINT_ALIAS_DESCRIPTION,
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based position in the full mux readout.",
        ),
    ]
    gate_count: Annotated[
        list[Annotated[int, Field(strict=True, ge=0)]],
        FieldMeta(
            label="CZ gates per shot",
            unit="count",
            description="Enter consecutive train lengths to measure phase per CZ.",
        ),
    ] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=3)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per point in one round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of times to accumulate each projection.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            decimals=3,
            description="Delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Raise ValueError for invalid tone calibration, pulses or train lengths."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        for drive, x90 in (
            (self.drive.qubit1, self.x90_qubit1),
            (self.drive.qubit2, self.x90_qubit2),
        ):
            if drive.ch != x90.ch:
                raise ValueError("CZ and X90 must use the same target generator")
            if not isinstance(x90.phase, (int, float)) or not np.isfinite(x90.phase):
                raise ValueError("X90 phase must be a finite scalar")
        if any(b - a != 1 for a, b in pairwise(self.gate_count)):
            raise ValueError("Gate counts must be consecutive and strictly increasing")
        return self


def restore_result(data: GroupedLoadData[LocalZHeraldedCfg]) -> LocalZHeraldedResult:
    """Load classified sums and original train counts; propagate missing-variable errors."""
    sums = data.variable("sums")
    return LocalZHeraldedResult(
        sums.axes[4].astype(np.int64), sums.z.astype(np.float64)
    )


class LocalZHeraldedExp(Experiment[LocalZHeraldedResult, LocalZHeraldedCfg]):
    """Herald and measure both targets without filtering any acquired shot."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZHeraldedResult, LocalZHeraldedCfg]] = (
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
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit1", unit=""
                        ),
                        VariableAxisSpec(
                            label="CZ gate count",
                            unit="",
                            field_name="gate_counts",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Projection X or Y",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Target Qubit1 or Qubit2",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
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
            result_type=LocalZHeraldedResult,
            cfg_type=LocalZHeraldedCfg,
            cfg_schema_version="1.0",
            tag="heralded/local_z",
            result_builder=restore_result,
        )
    )

    def analyze(
        self,
        source: RunRecord[LocalZHeraldedCfg, LocalZHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> LocalZHeraldedAnalysis:
        """Analyze a classified record; reject missing cfg, wrong shape or train axes."""
        del options, plots
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Local Z analysis requires saved source cfg")
        if not np.array_equal(data.gate_counts, cfg.gate_count):
            raise ValueError("Local Z count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_count), 3, 3, 3, 3):
            raise ValueError(
                "Expected target × X/Y × n × herald Q1/Q2 × terminal Q1/Q2 counts"
            )
        # Baseline for the focused red probe: only the existing gg estimate is available.
        return LocalZHeraldedAnalysis(
            (
                LocalZJointFit(reason="No shared slope"),
                LocalZJointFit(reason="No shared slope"),
            ),
            _gg_diagnostics(data.gate_counts, data.sums),
            (
                LocalZSpectatorComparison(reason="No spectator comparison"),
                LocalZSpectatorComparison(reason="No spectator comparison"),
            ),
        )


def _gg_diagnostics(
    ns: NDArray[np.int64], sums: NDArray[np.float64]
) -> tuple[LocalZGGFit, LocalZGGFit]:
    """Keep the ordinary count-weighted phase-line result on the gg slice."""
    gg = terminal_counts(sums, ("g", "g"))
    fits = []
    for t, target in enumerate(_TARGETS):
        group = count_quadratures(gg[t], target)
        x, y = group.values
        angles = [
            count_phase(float(a), float(b)).logical_phase_deg
            for a, b in zip(x, y, strict=True)
        ]
        if None in angles:
            reason = (
                "No accepted shots at one or more quadratures"
                if np.any(group.acquired == 0)
                else "Incomplete phase row"
            )
            fits.append(LocalZGGFit(None, None, None, None, None, reason))
            continue
        fit = fit_phase_line(
            ns.astype(np.float64),
            np.array(angles, dtype=np.float64),
            phase_variance(x, y, *group.variances),
        )
        fits.append(
            LocalZGGFit(
                fit.intercept_deg,
                fit.slope_deg,
                None if fit.slope_deg is None else -fit.slope_deg,
                fit.slope_error_deg,
                fit.fit_quality,
                fit.reason,
            )
        )
    return fits[0], fits[1]
