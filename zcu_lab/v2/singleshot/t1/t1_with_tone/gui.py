from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
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
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import SweepDefault, custom, scaled_path_seed
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
    experiment_tag: ClassVar[str | None] = (
        T1WithToneExp.AXES_SPEC.tag if T1WithToneExp.AXES_SPEC is not None else None
    )
    exp_cls = T1WithToneExp
    ExpCfg_cls: ClassVar[Any] = T1WithToneCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone: applies a π pulse and a simultaneous "
            "probe tone during the wait, sweeps the wait-plus-probe length, "
            "and classifies each shot in-program against the |g>/|e> IQ-cluster "
            "centres. Plots the ground / excited / other populations versus "
            "delay time with dual-transition-rate fits for T1 and T1_b. The "
            "fitted T1 value is written back to 't1_with_tone' in the parameter context. "
            "Runs on real hardware."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live parameter context. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. "
            "Optionally reads 'confusion_matrix' to readout-correct populations "
            "at analyze time; 't1_with_tone' or 't1' to seed the sweep stop "
            "(5*t1_with_tone when present, else 5*t1; fallback 500 us); "
            "'t1' seeds relax delay (5*t1; fallback 100 us); 'q_f' / "
            "'qub_ch' for the pi pulse; 'readout_f' or 'r_f' plus "
            "'res_ch' seed the probe tone; "
            "'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' for readout."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module, a probe-tone pulse module, and a "
            "readout module. Optional reset and optional init pulse (both "
            "disabled when no library entry exists)."
        ),
        typical_writeback=(
            "Proposes the fitted T1 relaxation time under the probe tone into "
            "parameter context 't1_with_tone' (us)."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. "
            "Use 'uniform=False' (default) to "
            "cluster points along the expected exponential decay while preserving "
            "the configured window and point count; use 'uniform=True' for a "
            "linear sweep. The probe-tone gain and frequency are set inside the "
            "probe_pulse module."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
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
