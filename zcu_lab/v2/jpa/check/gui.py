"""jpa/check GUI adapter — JPA pump off/on resonator comparison check.

Owns the jpa/check cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``
(``CheckExp``).
"""

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
    ModuleInit,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import res_freq_range
from zcu_lab.v2.jpa.check.core import CheckCfg, CheckExp, CheckResult

JpaCheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]

# Bring-up survey: ~101 readout-frequency points around the resonator. These
# are inspectable starting bounds, NOT safety certification — the operator must
# review device and sweep.
_JPA_CHECK_FREQ_EXPTS = 101


@dataclass
class JpaCheckAnalyzeResult(AnalyzeResultBase):
    pass


class JpaCheckAdapter(
    BaseAdapter[CheckCfg, JpaCheckRunResult, JpaCheckAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = CheckExp.persistence_tag()
    exp_cls = CheckExp
    ExpCfg_cls: ClassVar[Any] = CheckCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                },
            )
            .relax_delay(0.5)
            .device(
                "jpa_rf_dev",
                label="JPA RF device",
                default="",
                required=False,
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=_JPA_CHECK_FREQ_EXPTS),
            )
            .reps(1000)
            .rounds(5)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, CheckExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaCheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaCheckAnalyzeResult:
        CheckExp().analyze(req.run_result, None, plots=plots)
        return JpaCheckAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_check_{time.strftime('%m%d')}"
