from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import CfgSchema

T1RunResult: TypeAlias = T1Result


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")] = False


@dataclass
class T1AnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_err: float
    figure: Figure


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    exp_cls = T1Exp
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T1 energy relaxation: applies a pi pulse to excite the qubit, "
            "then sweeps a wait delay before readout and fits the exponential "
            "decay to extract T1. Runs on real hardware. Run after a pi pulse "
            "has been calibrated (amplitude/length Rabi)."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional, seeding defaults): 't1' — "
            "prior T1 estimate (us); the delay sweep spans up to 5*t1 and "
            "relax_delay defaults to 5*t1 (both fallback ~100 us). The pi "
            "pulse pulls 'q_f' (~2000–6000 MHz) and 'qub_ch'. Readout pulls "
            "'r_f' (~4000–8000 MHz), 'res_ch' / 'ro_ch', and 'timeFly' for the "
            "readout trigger offset (~0–1 us)."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module — references a calibrated library "
            "pi pulse ('pi_amp' / 'pi_len') when present, else a blank inline "
            "pulse — and a readout module (calibrated 'readout_dpm' / "
            "'readout_rf' / 'readout' / 'res_readout', else a blank "
            "pulse-readout referencing 'ro_waveform' when present). Optional "
            "reset references a library reset ('reset_bath' / 'reset_10' / "
            "'reset_120') when present, else stays disabled."
        ),
        typical_writeback=(
            "Proposes the fitted T1 relaxation time into MetaDict 't1' (us). "
            "No ModuleLibrary writeback."
        ),
        recommended=(
            "Analysis defaults to a single-exponential fit; enable "
            "dual-exponential only when the decay clearly shows two timescales "
            "(a fast component on top of the slow relaxation). A delay sweep "
            "reaching ~5*T1 lets the decay flatten so the fit is "
            "well-constrained; with no prior 't1' it spans 0–100 us. "
            "Set 'uniform=False' to cluster more points on the exponential "
            "decay while keeping the configured start/stop window."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset("reset", optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(
                scaled_md("t1", factor=5.0, fallback_value=100.0),
            )
            .sweep(
                "length",
                label="Delay (us)",
                default=SweepDefault(
                    start=0.0,
                    # The fallback is the fully scaled 5 * 100 us window.
                    stop=scaled_md("t1", factor=5.0, fallback_value=500.0),
                    expts=101,
                ),
            )
            .bool(
                "uniform",
                label="Uniform (linear) sweep",
                default=True,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T1Cfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("uniform", None)
        return super().build_exp_cfg(cfg_raw, req)

    def _uniform(self, raw_cfg: dict[str, object]) -> bool:
        value = raw_cfg.get("uniform", True)
        if not isinstance(value, bool):
            raise ValueError(f"'uniform' must be a bool, got {type(value).__name__}")
        return value

    def run(self, req: RunRequest, schema: CfgSchema) -> T1RunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        return T1Exp().run(soc, soccfg, cfg, uniform=self._uniform(raw_cfg))

    def analyze(
        self, req: AnalyzeRequest[T1RunResult, T1AnalyzeParams]
    ) -> T1AnalyzeResult:
        params = req.analyze_params
        t1, t1_err, fig = T1Exp().analyze(
            req.run_result,
            dual_exp=params.dual_exp,
        )
        return T1AnalyzeResult(t1=t1, t1_err=t1_err, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[T1RunResult, T1AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="t1",
                description="T1 relaxation time (us)",
                proposed_value=result.t1,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
