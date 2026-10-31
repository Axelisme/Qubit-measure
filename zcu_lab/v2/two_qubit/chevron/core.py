"""Common MHz frequency scan with fixed signed gains and relative phase."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, GroupedLoadData, VariableAxisSpec, VariableSpec, VariableZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from zcu_tools.program.v2 import SweepCfg

from .._drive import DualLineCfg
from .._rabi_analysis import RabiAnalysis, fit_duration_rates, joint_return, plot_duration_rates, plot_return_data
from .._rabi_scan import DurationScanCfg, RabiScanResult, acquire_counts


class ChevronCfg(DurationScanCfg):
    """Common MHz frequency scan with fixed signed gains and relative phase.

    Both frequencies change together. Gain ratio, phases and source waveforms remain fixed.
    Optional reset precedes preparation. F1/F2 classifiers require GE Post radii.
    """

    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz", label="Common drive frequency",
            description="Set the common drive frequency sweep endpoints in MHz; review the scan window before Run.",
        ),
    ]

    @model_validator(mode="after")
    def validate_coordinate(self) -> Self:
        """Require finite requested frequency coordinates."""
        if not np.all(np.isfinite(sweep2array(self.frequency))):
            raise ValueError("Common drive frequency coordinates must be finite")
        return self


@dataclass(frozen=True)
class ChevronResult(RabiScanResult):
    """Common drive frequency coordinates, durations in us and (Nc,Nd,3,3) counts.

    Terminal class axes are F1/F2, each g/e/other. Unmeasured points are NaN.
    Counts are float64 totals, not normalized probabilities.
    """


def acquire_scan(config: ChevronCfg, *, context: RunContext, heralded: bool) -> RabiScanResult:
    """Acquire this scan with optional herald; no input mutation or host round loop.

    Coordinate transforms preserve signed gains, timing and source waveforms.
    The shared duration sequence prepares B11, drives both lines, applies double
    Xπ contrast and emits terminal readout. Check context.scope.stop for errors.
    """
    coordinates = sweep2array(config.frequency)
    drives = [
        DualLineCfg(
            f1=config.drive.f1.with_updates(freq=coordinate),
            f2=config.drive.f2.with_updates(freq=coordinate),
        )
        for coordinate in coordinates
    ]
    return acquire_counts(config, coordinates, drives, context=context, heralded=heralded)


def analyze_scan(
    cfg: ChevronCfg, result: RabiScanResult, *, plots: Plots, heralded: bool
) -> RabiAnalysis:
    """Analyze count-derived P00 and duration-slice rates.

    Heralded input selects herald (g,g) and additionally plots its nine-class
    distribution. Other remains in the terminal denominator. Empty traces are
    unresolved. Saved cfg determines axes; the current Form/library is unused.
    """
    answer = joint_return(cfg, result, sweep2array(cfg.frequency), heralded=heralded)
    plot_return_data(
        result, answer, coordinate_label="Common drive frequency (MHz)",
        map_name="chevron", plots=plots,
    )
    rates = fit_duration_rates(result, answer)
    plot_duration_rates(
        result, rates, coordinate_label="Common drive frequency (MHz)", plots=plots
    )
    return RabiAnalysis(answer, rates)


def restore_result(data: GroupedLoadData[ChevronCfg]) -> ChevronResult:
    """Restore summed counts; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ChevronResult(
        coordinates=sums.axes[3], durations=sums.axes[2], sums=sums.z.astype(np.float64)
    )


class ChevronExp(Experiment[ChevronResult, ChevronCfg]):
    """Acquire and persist the ordinary chevron count scan."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ChevronResult, ChevronCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec("Drive duration", "s", field_name="durations", scale=1e-6),
                    VariableAxisSpec("Common drive frequency", "Hz", field_name="coordinates", scale=1e6),
                ),
                z=VariableZSpec(field_name="sums", label="Terminal class counts", unit="", dtype=np.float64),
            ),
        ),
        result_type=ChevronResult, cfg_type=ChevronCfg,
        cfg_schema_version="4.0", tag="two_qubit/chevron", result_builder=restore_result,
    )

    def run(self, config: ChevronCfg, *, context: RunContext) -> ChevronResult:
        """Acquire optional reset → B11 → dual drive → double Xπ → terminal.

        Return unchanged counts, including completed rounds on cancellation.
        Rounds accumulate at acquisition. Check context.scope.stop for errors.
        """
        data = acquire_scan(config, context=context, heralded=False)
        return ChevronResult(data.coordinates, data.durations, data.sums)

    def analyze(
        self, source: RunRecord[ChevronCfg, ChevronResult], options: None, *, plots: Plots
    ) -> RabiAnalysis:
        """Use saved count axes and cfg to plot return and fit eligible traces.

        Missing cfg or mismatched axes raises ValueError. Failed fits report a
        reason, not zero rates. No IQ classification or automatic write occurs.
        """
        del options
        if source.cfg is None:
            raise ValueError("Rabi analysis requires saved cfg; current Form/library is not a substitute")
        return analyze_scan(source.cfg, source.result, plots=plots, heralded=False)
