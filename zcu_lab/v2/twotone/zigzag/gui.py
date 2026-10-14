"""GUI adapter for the plain ZigZag repetition experiment."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import scaled_path_seed

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.zigzag import (
    add_gate_modules,
    add_repeat_fields,
)
from zcu_lab.v2.twotone.zigzag.core import ZigZagCfg, ZigZagExp, ZigZagResult

ZigZagRunResult: TypeAlias = RunRecord[ZigZagCfg, ZigZagResult]


class ZigZagAdapter(BaseAdapter[ZigZagCfg, ZigZagRunResult]):
    experiment_tag: ClassVar[str | None] = ZigZagExp.persistence_tag()
    exp_cls = ZigZagExp
    ExpCfg_cls: ClassVar[Any] = ZigZagCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        builder = add_gate_modules(
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
        ).relax_delay(scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5))
        return (
            add_repeat_fields(builder, n_times=10)
            .bool(
                "reset_phase_cycle",
                label="Cycle reset pi phase (0/180)",
                default=False,
                tooltip="Alternate the final reset pulse by 180 degrees between averaging sweeps; requires even reps. Does not correct population or repeated-gate errors.",
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> ZigZagRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ZigZagExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_zigzag_{time.strftime('%m%d')}"
