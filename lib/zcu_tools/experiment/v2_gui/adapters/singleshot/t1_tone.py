from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import (
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    SweepDefault,
    custom,
    md_eval_scaled,
    md_has_key,
    scaled_md,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    EvalValue,
)

from ._shared import read_ge_centers, readout_probe_freq

# Domain T1WithToneExp.analyze returns (t1, t1_b, fig). The T1 with tone value
# (``t1_with_tone``) is written back to the MetaDict (key ``t1_with_tone``,
# matching single_qubit.md:3079).
SsT1ToneRunResult: TypeAlias = T1WithToneResult


def _sweep_stop_default(ctx: ExpContext) -> float | EvalValue:
    key = "t1_with_tone" if md_has_key(ctx, "t1_with_tone") else "t1"
    return md_eval_scaled(ctx, key, factor=5.0, fallback=100.0)


@dataclass
class SsT1ToneAnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_b: float
    figure: Figure


class SsT1ToneAdapter(
    BaseAdapter[
        T1WithToneCfg,
        SsT1ToneRunResult,
        SsT1ToneAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = T1WithToneExp
    ExpCfg_cls: ClassVar[Any] = T1WithToneCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1-with-tone: applies a π pulse and a simultaneous "
            "probe tone during the wait, sweeps the wait-plus-probe length, "
            "and classifies each shot in-program against the |g>/|e> IQ-cluster "
            "centres. Plots the ground / excited / other populations versus "
            "delay time with dual-transition-rate fits for T1 and T1_b. The "
            "fitted T1 value is written back to 't1_with_tone' in the MetaDict. "
            "Runs on real hardware."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run "
            "fast-fails if any is missing. "
            "Optionally reads 'confusion_matrix' to readout-correct populations "
            "at analyze time; 't1_with_tone' or 't1' to seed the sweep stop "
            "(default 5*t1, fallback 100 us); 'q_f' / 'qub_ch' for the pi "
            "pulse; 'readout_f' or 'r_f' plus 'res_ch' seed the probe tone; "
            "'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' for readout."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module, a probe-tone pulse module, and a "
            "readout module. Optional reset and optional init pulse (both "
            "disabled when no library entry exists)."
        ),
        typical_writeback=(
            "Proposes the fitted T1 relaxation time under the probe tone into "
            "MetaDict 't1_with_tone' (us)."
        ),
        recommended=(
            "Run after 'singleshot/ge'. Use 'uniform=False' (default) for "
            "log-spaced delays (better sampling of the decay); 'uniform=True' "
            "for a linear sweep. The probe-tone gain and frequency are set "
            "inside the probe_pulse module."
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
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
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
            .bool("uniform", label="Uniform (linear) sweep", default=False)
            .reps(1000)
            .rounds(10)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: dict[str, object], req: RunRequest
    ) -> T1WithToneCfg:
        # Pop ``uniform`` before lowering — it is not part of T1WithToneCfg.
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("uniform", None)
        return req.ml.make_cfg(cfg_raw, T1WithToneCfg)

    def _uniform(self, raw_cfg: dict[str, object]) -> bool:
        value = raw_cfg.get("uniform", False)
        if not isinstance(value, bool):
            raise ValueError(f"'uniform' must be a bool, got {type(value).__name__}")
        return value

    def run(self, req: RunRequest, schema: CfgSchema) -> SsT1ToneRunResult:
        # Override standard run: domain run needs GE centres + uniform kwarg.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        uniform = self._uniform(raw_cfg)
        return T1WithToneExp().run(
            soc, soccfg, cfg, g_center, e_center, radius, uniform=uniform
        )

    def analyze(
        self, req: AnalyzeRequest[SsT1ToneRunResult, NoAnalyzeParams]
    ) -> SsT1ToneAnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        confusion = req.md.get("confusion_matrix")
        t1, t1_b, fig = T1WithToneExp().analyze(
            req.run_result, confusion_matrix=confusion
        )
        return SsT1ToneAnalyzeResult(t1=t1, t1_b=t1_b, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsT1ToneRunResult, SsT1ToneAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``t1_with_tone`` per single_qubit.md:3079.
        return [
            MetaDictWriteback(
                target_name="t1_with_tone",
                description="T1 with probe tone (us)",
                proposed_value=req.analyze_result.t1,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_t1_tone_{time.strftime('%m%d')}"
