from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
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
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

T2RamseyRunResult: TypeAlias = T2RamseyResult


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")]


@dataclass
class T2RamseyAnalyzeResult(AnalyzeResultBase):
    t2r: float
    t2r_err: float
    detune: float
    figure: Figure


class T2RamseyAdapter(
    BaseAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    exp_cls = T2RamseyExp
    ExpCfg_cls: ClassVar[Any] = T2RamseyCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "T2 Ramsey: two pi/2 pulses separated by a swept free-evolution "
                "delay; fits the decaying fringe to extract the Ramsey coherence "
                "time T2* and the qubit-drive detuning. Runs on real hardware. Run "
                "after a pi/2 pulse is calibrated; a small deliberate detuning "
                "produces the fringe whose frequency reads out the true qubit "
                "frequency offset."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional, seeding defaults): 't2r' — "
                "prior T2-Ramsey estimate (us); the delay sweep spans up to 4*t2r "
                "(fallback ~20 us). 't1' seeds relax_delay (5*t1, fallback ~100 "
                "us). The pi/2 pulse pulls 'q_f' (~2000–6000 MHz) and 'qub_ch'. "
                "Readout pulls 'r_f' (~4000–8000 MHz), 'res_ch' / 'ro_ch', and "
                "'timeFly' for the readout trigger offset (~0–1 us)."
            ),
            expects_ml=(
                "Needs a qubit pi/2-pulse module — references a calibrated library "
                "entry ('pi2_amp' / 'pi2_len', degrading to 'pi_amp' / 'pi_len') "
                "when present, else a blank inline pulse — and a readout module "
                "(calibrated 'readout_dpm' / 'readout_rf' / 'readout' / "
                "'res_readout', else a blank pulse-readout referencing "
                "'ro_waveform' when present). Optional reset references a library "
                "reset ('reset_bath' / 'reset_10' / 'reset_120') when present."
            ),
            typical_writeback=(
                "Proposes the fitted T2-Ramsey time into MetaDict 't2r' (us). The "
                "fitted detuning is computed and shown but is NOT written back. No "
                "ModuleLibrary writeback."
            ),
            recommended=(
                "Analysis defaults to fitting the fringe (returning both T2* and "
                "the detuning) — keep it on when you ran with a deliberate "
                "detuning so an oscillation is visible; turn it off for a pure "
                "decay fit (detune reported 0) when driving on resonance. A delay "
                "sweep reaching a few T2* captures enough fringe periods and decay."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "pi2_pulse": make_pulse_module_spec(),
                        "readout": make_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"length": SweepSpec(label="Delay (us)")},
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t2r", factor=4.0, fallback=20.0)
        relax_delay = proper_relax(ctx)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=relax_delay)
            .role("modules.pi2_pulse", "pi2_pulse")
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .set_sweep(
                "sweep.length", SweepValue(start=0.0, stop=sweep_stop, expts=101)
            )
            .build()
        )

    def get_analyze_params(
        self, result: T2RamseyRunResult, ctx: ExpContext
    ) -> T2RamseyAnalyzeParams:
        return T2RamseyAnalyzeParams(fit_fringe=True)

    def analyze(
        self, req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeParams]
    ) -> T2RamseyAnalyzeResult:
        params = req.analyze_params
        t2r, t2r_err, detune, _, fig = T2RamseyExp().analyze(
            req.run_result,
            fit_fringe=params.fit_fringe,
        )
        return T2RamseyAnalyzeResult(
            t2r=t2r, t2r_err=t2r_err, detune=detune, figure=fig
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="t2r",
                description="T2 Ramsey time (us)",
                proposed_value=result.t2r,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2ramsey_{time.strftime('%m%d')}"
