from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.cfg import (
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.reset.rabi_check.core import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)

RabiCheckRunResult: TypeAlias = RunRecord[RabiCheckCfg, RabiCheckResult]


@dataclass
class RabiCheckAnalyzeResult(AnalyzeResultBase):
    frequency_cycles_per_gain: float
    before_amplitude: float
    after_amplitude: float
    relative_contrast: float
    before_phase_deg: float | None
    after_phase_deg: float | None
    phase_difference_deg: float | None
    reset_residual_amplitude: float
    reset_offset: float
    after_second_harmonic_amplitude: float
    before_residual_rms: float
    reset_residual_rms: float
    after_residual_rms: float


class RabiCheckAdapter(
    BaseAdapter[
        RabiCheckCfg, RabiCheckRunResult, RabiCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    """Rabi-amplitude check for any reset type (single-tone / two-pulse / bath).

    Sweeps the initialisation pulse gain and records three signal branches in
    parallel (without tested_reset / with tested_reset / with tested_reset + rabi_pulse),
    fitting relative Rabi contrast and residual input dependence. Averaged IQ
    diagnostics do not determine reset fidelity or a unique reset channel.
    """

    experiment_tag: ClassVar[str | None] = RabiCheckExp.persistence_tag()

    exp_cls = RabiCheckExp
    ExpCfg_cls: ClassVar[Any] = RabiCheckCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, analysis=AnalysisMode.FIT, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("rabi_pulse", role_id="pi_pulse", label="Rabi Pulse")
            .reset("tested_reset", role_id="reset", label="Tested Reset")
            .readout()
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .reps(1000)
            .rounds(10)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RabiCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = RabiCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[RabiCheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> RabiCheckAnalyzeResult:
        fit = RabiCheckExp().analyze(req.run_result, None, plots=plots)
        return RabiCheckAnalyzeResult(
            frequency_cycles_per_gain=fit.frequency,
            before_amplitude=fit.before.amplitude,
            after_amplitude=fit.after.amplitude,
            relative_contrast=fit.contrast_ratio,
            before_phase_deg=fit.before.phase_deg,
            after_phase_deg=fit.after.phase_deg,
            phase_difference_deg=fit.phase_difference_deg,
            reset_residual_amplitude=fit.reset.amplitude,
            reset_offset=fit.reset.offset,
            after_second_harmonic_amplitude=fit.after.second_harmonic_amplitude,
            before_residual_rms=fit.before.residual_rms,
            reset_residual_rms=fit.reset.residual_rms,
            after_residual_rms=fit.after.residual_rms,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_reset_check_{time.strftime('%m%d')}"
