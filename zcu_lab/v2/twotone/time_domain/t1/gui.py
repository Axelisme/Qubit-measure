from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.twotone.time_domain.t1.core import T1Analysis
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgDefinition
from zcu_lab.v2._support.measure.seeds import SweepDefault
from zcu_lab.v2._support.measure.seeds import scaled_md
from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")] = False
    skip: Annotated[int, ParamMeta(label="Skip leading points")] = 0


@dataclass(frozen=True)
class T1AnalyzeResult:
    """GUI summary over the unchanged, typed core analysis."""

    analysis: T1Analysis

    @property
    def t1(self) -> float:
        return self.analysis.t1

    @property
    def t1_err(self) -> float:
        return self.analysis.t1_err

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "t1": self.analysis.t1,
            "t1_err": self.analysis.t1_err,
            "t1b": self.analysis.t1b,
            "t1b_err": self.analysis.t1b_err,
            "fit_quality": fit_quality_summary(self.analysis.fit_quality),
        }


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

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> T1RunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = T1Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[T1RunResult, T1AnalyzeParams], *, plots: Plots
    ) -> T1AnalyzeResult:
        params = req.analyze_params
        if params.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        analysis = T1Exp().analyze(
            req.run_result,
            T1AnalyzeOptions(dual_exp=params.dual_exp, skip=params.skip),
            plots=plots,
        )
        return T1AnalyzeResult(analysis)

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

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
