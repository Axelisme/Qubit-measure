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
from zcu_tools.gui.app.measure.seeds import custom, scaled_path_seed
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
    readout_probe_freq,
    readout_probe_freq_range,
)
from zcu_lab.v2.singleshot.mist.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqDepExp,
    FreqResult,
)

MistFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class MistFreqAnalyzeResult(AnalyzeResultBase):
    pass


class MistFreqAdapter(
    BaseAdapter[FreqCfg, MistFreqRunResult, MistFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqDepExp.persistence_tag()
    exp_cls = FreqDepExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

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
            .relax_delay(scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5))
            .sweep(
                "freq",
                label="Probe freq (MHz)",
                default=custom(
                    lambda ctx: readout_probe_freq_range(ctx, 51),
                    description="readout probe frequency range",
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
            .reps(10000)
            .rounds(1)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> MistFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistFreqAnalyzeResult:
        options = FreqAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        FreqDepExp().analyze(req.run_result, options, plots=plots)
        return MistFreqAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_freq_{time.strftime('%m%d')}"
