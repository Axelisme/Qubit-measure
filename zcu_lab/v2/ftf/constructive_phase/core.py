"""Thermal B11 relative-phase × duration experiment."""

from typing import Annotated, Self

import numpy as np
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from .._drive import DualLineCfg
from .._rabi_scan import (
    DurationScanCfg,
    RabiScanResult,
    joint_return,
    plot_return_data,
    run_duration_scan,
)
from .analysis import ConstructivePhaseAnalysis, analyze_constructive_phase


class ConstructivePhaseCfg(DurationScanCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency.

    Each point replaces F2.phase with F1.phase + relative_phase. Signed gains,
    their ratio, waveform styles and delays remain unchanged. No parasitic
    cancellation, amplitude-ratio search or heralding is performed.
    """

    relative_phase: Annotated[
        SweepCfg, FieldMeta(unit="deg", label="F2 relative to F1 phase")
    ]

    @model_validator(mode="after")
    def validate_phase(self) -> Self:
        """Require finite requested relative phases."""
        if not np.all(np.isfinite(sweep2array(self.relative_phase))):
            raise ValueError("Relative phases must be finite")
        return self


class ConstructivePhaseExp(PersistableExperiment[RabiScanResult, ConstructivePhaseCfg]):
    """Stateless relative-phase acquisition and native persistence."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", IDENTITY, dtype=np.int64),
            Axis("durations", "Drive duration", "s", US_TO_S),
            Axis("coordinates", "F2 relative to F1 phase", "deg", IDENTITY),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=RabiScanResult,
        cfg_type=ConstructivePhaseCfg,
        cfg_schema_version="1.0",
        tag="ftf/constructive_phase",
    )

    def run(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> RabiScanResult:
        """Acquire B11 → dual drive → double Xπ → mux for phase/duration/round.

        Preserve all raw shots and full tones, including partial cancellation.
        Caller must check context.scope.stop.raise_if_error(). No input mutation.
        """
        phases = sweep2array(config.relative_phase)
        drives = [
            DualLineCfg(
                f1=config.drive.f1.model_copy(deep=True),
                f2=config.drive.f2.with_updates(
                    phase=float(config.drive.f1.phase) + phase
                ),
            )
            for phase in phases
        ]
        return run_duration_scan(config, phases, drives, context=context)

    def analyze(
        self,
        source: RunRecord[ConstructivePhaseCfg, RabiScanResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Classify complete points and plot return using saved cfg only.

        Fit complete duration traces with the undamped cosine, excluding rates
        below 1/saved duration span. Fit rate²=A+B cosφ+C sinφ and return the
        maximum-rate relative phase, or a reason for unresolved/degenerate fits.
        No contrast/sigma, frequency/sigma or residual quality gate is applied.
        Missing source cfg, invalid raw or malformed axes raise ValueError.
        Incomplete points keep raw but are not classified or replaced by zero.
        """
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        answer = joint_return(
            source.cfg, source.result, sweep2array(source.cfg.relative_phase)
        )
        plot_return_data(
            source.result,
            answer,
            coordinate_label="F2 relative to F1 phase (deg)",
            map_name="interference",
            plots=plots,
        )
        return analyze_constructive_phase(source.result, answer, plots=plots)
