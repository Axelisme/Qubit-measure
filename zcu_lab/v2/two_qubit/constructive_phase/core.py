"""Relative F2 minus F1 phase scan in degrees at fixed common frequency."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
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
from zcu_tools.program.v2 import SweepCfg

from .._drive import DualLineCfg
from .._rabi_analysis import joint_return, plot_return_data
from .._rabi_scan import DurationScanCfg, RabiScanResult, acquire_counts
from .analysis import ConstructivePhaseAnalysis, analyze_constructive_phase


class ConstructivePhaseCfg(DurationScanCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency.

    Each coordinate sets F2.phase to F1.phase plus the requested phase. Signed gains and source waveforms remain fixed.
    Optional reset precedes preparation. F1/F2 classifiers require GE Post radii.
    """

    relative_phase: Annotated[
        SweepCfg,
        FieldMeta(
            unit="deg", label="F2 relative to F1 phase",
            description="Set the f2 relative to f1 phase sweep endpoints in deg; review the scan window before Run.",
        ),
    ]

    @model_validator(mode="after")
    def validate_coordinate(self) -> Self:
        """Require finite requested relative_phase coordinates."""
        if not np.all(np.isfinite(sweep2array(self.relative_phase))):
            raise ValueError("F2 relative to F1 phase coordinates must be finite")
        return self


@dataclass(frozen=True)
class ConstructivePhaseResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are F1/F2, each g/e/other. Unmeasured points are NaN.
    Counts are float64 totals, not normalized probabilities.
    """


def acquire_scan(config: ConstructivePhaseCfg, *, context: RunContext, heralded: bool) -> RabiScanResult:
    """Acquire this scan with optional herald; no input mutation or host round loop.

    Coordinate transforms preserve signed gains, timing and source waveforms.
    The shared duration sequence prepares B11, drives both lines, applies double
    Xπ contrast and emits terminal readout. Check context.scope.stop for errors.
    """
    coordinates = sweep2array(config.relative_phase)
    drives = [
        DualLineCfg(
            f1=config.drive.f1.model_copy(deep=True),
            f2=config.drive.f2.with_updates(phase=float(config.drive.f1.phase) + coordinate),
        )
        for coordinate in coordinates
    ]
    return acquire_counts(config, coordinates, drives, context=context, heralded=heralded)


def analyze_scan(
    cfg: ConstructivePhaseCfg, result: RabiScanResult, *, plots: Plots, heralded: bool
) -> ConstructivePhaseAnalysis:
    """Analyze count-derived P00 and the constructive rate² maximum.

    Heralded input selects herald (g,g) and additionally plots its nine-class
    distribution. Other remains in the terminal denominator. Empty traces are
    unresolved. Saved cfg determines axes; the current Form/library is unused.
    """
    answer = joint_return(cfg, result, sweep2array(cfg.relative_phase), heralded=heralded)
    plot_return_data(
        result, answer, coordinate_label="F2 relative to F1 phase (deg)",
        map_name="interference", plots=plots,
    )
    return analyze_constructive_phase(result, answer, plots=plots)


def restore_result(data: GroupedLoadData[ConstructivePhaseCfg]) -> ConstructivePhaseResult:
    """Restore summed counts; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ConstructivePhaseResult(
        coordinates=sums.axes[3], durations=sums.axes[2], sums=sums.z.astype(np.float64)
    )


class ConstructivePhaseExp(Experiment[ConstructivePhaseResult, ConstructivePhaseCfg]):
    """Acquire and persist the ordinary constructive_phase count scan."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ConstructivePhaseResult, ConstructivePhaseCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec("Drive duration", "s", field_name="durations", scale=1e-6),
                    VariableAxisSpec("F2 relative to F1 phase", "deg", field_name="coordinates", scale=1.0),
                ),
                z=VariableZSpec(field_name="sums", label="Terminal class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=ConstructivePhaseResult, cfg_type=ConstructivePhaseCfg,
        cfg_schema_version="4.0", tag="two_qubit/constructive_phase", result_builder=restore_result,
    )

    def run(self, config: ConstructivePhaseCfg, *, context: RunContext) -> ConstructivePhaseResult:
        """Acquire optional reset → B11 → dual drive → double Xπ → terminal.

        Return unchanged counts, including completed rounds on cancellation.
        Rounds accumulate at acquisition. Check context.scope.stop for errors.
        """
        data = acquire_scan(config, context=context, heralded=False)
        return ConstructivePhaseResult(data.coordinates, data.durations, data.sums)

    def analyze(
        self, source: RunRecord[ConstructivePhaseCfg, ConstructivePhaseResult], options: None, *, plots: Plots
    ) -> ConstructivePhaseAnalysis:
        """Use saved count axes and cfg to plot return and fit eligible traces.

        Missing cfg or mismatched axes raises ValueError. Failed fits report a
        reason, not zero rates. No IQ classification or automatic write occurs.
        """
        del options
        if source.cfg is None:
            raise ValueError("Rabi analysis requires saved cfg; current Form/library is not a substitute")
        return analyze_scan(source.cfg, source.result, plots=plots, heralded=False)
