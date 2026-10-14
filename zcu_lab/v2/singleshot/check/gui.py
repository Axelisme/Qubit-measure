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
from zcu_tools.gui.app.measure.seeds import scaled_path_seed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_shared import read_ge_centers
from zcu_lab.v2.singleshot.check.core import (
    CheckAnalyzeOptions,
    CheckCfg,
    CheckExp,
    CheckResult,
)

CheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


@dataclass
class CheckAnalyzeResult(AnalyzeResultBase):
    # Classification scatter is published through Plots; no numeric writeback.
    pass


class CheckAdapter(
    BaseAdapter[CheckCfg, CheckRunResult, CheckAnalyzeResult, NoAnalyzeParams]
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
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse(
                "probe_pulse",
                role_id="qub_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
            )
            .readout()
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .int("shots", label="Shots", default=5000)
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    # Standard run path (BaseAdapter.run): CheckExp.run(soc, soccfg, cfg) needs no
    # centres — the trio is an ANALYZE input, read from md in analyze().

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> CheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = CheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> CheckAnalyzeResult:
        # The check classifies the scatter against the GE centres — read the trio
        # from md (fast-fail if the upstream 'singleshot/ge' writeback is absent).
        g_center, e_center, radius = read_ge_centers(req.parameters)
        CheckExp().analyze(
            req.run_result, CheckAnalyzeOptions(g_center, e_center, radius), plots=plots
        )
        return CheckAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_check_{time.strftime('%m%d')}"
