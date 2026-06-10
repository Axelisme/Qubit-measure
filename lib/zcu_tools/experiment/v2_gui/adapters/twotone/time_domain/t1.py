from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

T1RunResult: TypeAlias = T1Result


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")]


@dataclass
class T1AnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_err: float
    figure: Figure


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    exp_cls = T1Exp
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
                "well-constrained; with no prior 't1' it spans 0–100 us."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "pi_pulse": make_pulse_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Delay (us)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t1", factor=5.0, fallback=100.0)
        relax_delay = proper_relax(ctx)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=relax_delay)
            .role("modules.pi_pulse", "pi_pulse")
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .set_sweep(
                "sweep.length", SweepValue(start=0.0, stop=sweep_stop, expts=101)
            )
            .build()
        )

    def get_analyze_params(
        self, result: T1RunResult, ctx: ExpContext
    ) -> T1AnalyzeParams:
        return T1AnalyzeParams(dual_exp=False)

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
