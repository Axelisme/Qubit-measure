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
    ModuleInit,
)
from zcu_tools.gui.cfg import (
    SweepValue,
)

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import qub_freq_range
from zcu_lab.v2.twotone.power_dep.core import PowerCfg, PowerExp, PowerResult

PowerDepRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


class PowerDepAdapter(BaseAdapter[PowerCfg, PowerDepRunResult]):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> PowerDepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                locked={"freq": 0.0, "gain": 0.0},
            )
            .readout()
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.001, stop=1.0, expts=101),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=qub_freq_range(expts=201),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"
