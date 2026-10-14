from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
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
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_tools.gui.app.measure.seeds import SweepDefault, custom, scaled_path_seed
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
    readout_probe_freq,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.core import (
    T1WithToneAnalyzeOptions,
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneResult,
)

# The numeric t1 is written to the t1_with_tone parameter context key.
SsT1ToneRunResult: TypeAlias = RunRecord[T1WithToneCfg, T1WithToneResult]


def _sweep_stop_default(_ctx: SessionEnv) -> EvalValue:
    return EvalValue("5.0 * $qubit.ext.t1_with_tone")


@dataclass
class SsT1ToneAnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_b: float


class SsT1ToneAdapter(
    BaseAdapter[
        T1WithToneCfg,
        SsT1ToneRunResult,
        SsT1ToneAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = T1WithToneExp.persistence_tag()
    exp_cls = T1WithToneExp
    ExpCfg_cls: ClassVar[Any] = T1WithToneCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .pulse(
                "probe_pulse",
                role_id="res_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="readout probe frequency",
                    )
                },
            )
            .readout()
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0)
            )
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    stop=custom(
                        _sweep_stop_default,
                        description="T1-with-tone delay stop",
                    ),
                    expts=101,
                ),
            )
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("$qubit.ext.readout_cal.g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("$qubit.ext.readout_cal.e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("$qubit.ext.readout_cal.radius"),
            )
            .bool("uniform", label="Uniform (linear) sweep", default=False)
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
    ) -> SsT1ToneRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, T1WithToneExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1ToneRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1ToneAnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        confusion = read_confusion_matrix(req.parameters)
        result = T1WithToneExp().analyze(
            req.run_result,
            T1WithToneAnalyzeOptions(confusion_matrix=confusion),
            plots=plots,
        )
        return SsT1ToneAnalyzeResult(t1=result.t1, t1_b=result.t1_b)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsT1ToneRunResult, SsT1ToneAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``t1_with_tone`` per single_qubit.md:3079.
        return [
            ParameterWriteback(
                target_name="t1_with_tone",
                description="T1 with probe tone (us)",
                proposed_value=req.analyze_result.t1,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_tone_{time.strftime('%m%d')}"
