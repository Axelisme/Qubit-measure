from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Sequence,
    TypeAlias,
)

from zcu_tools.experiment.v2.twotone.time_domain.t2echo import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
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

T2EchoRunResult: TypeAlias = T2EchoResult


@dataclass
class T2EchoAnalyzeParams:
    fit_method: Annotated[Literal["fringe", "decay"], ParamMeta(label="Fit method")]


@dataclass
class T2EchoAnalyzeResult(AnalyzeResultBase):
    t2e: float
    t2e_err: float
    figure: Figure


class T2EchoAdapter(
    BaseAdapter[
        T2EchoCfg,
        T2EchoRunResult,
        T2EchoAnalyzeResult,
        T2EchoAnalyzeParams,
    ]
):
    exp_cls = T2EchoExp
    ExpCfg_cls: ClassVar[Any] = T2EchoCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "T2 echo (spin/Hahn echo): a pi/2 – delay – pi – delay – pi/2 "
                "sequence with the total free-evolution delay swept; the "
                "refocusing pi pulse cancels low-frequency dephasing, and the "
                "decay fit yields the echo coherence time T2E. Runs on real "
                "hardware. Run after both pi and pi/2 pulses are calibrated; T2E "
                "is typically longer than the Ramsey T2*."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional, seeding defaults): 't2e' — "
                "prior T2-echo estimate (us); the total-delay sweep spans up to "
                "4*t2e (fallback ~20 us). 't1' seeds relax_delay (5*t1, fallback "
                "~100 us). The pi and pi/2 pulses pull 'q_f' (~2000–6000 MHz) and "
                "'qub_ch'. Readout pulls 'r_f' (~4000–8000 MHz), 'res_ch' / "
                "'ro_ch', and 'timeFly' for the readout trigger offset (~0–1 us)."
            ),
            expects_ml=(
                "Needs both a qubit pi-pulse module (prefers library 'pi_amp' / "
                "'pi_len') and a pi/2-pulse module (prefers 'pi2_amp' / 'pi2_len', "
                "degrading to 'pi_amp' / 'pi_len'); each falls back to a blank "
                "inline pulse. Plus a readout module (calibrated 'readout_dpm' / "
                "'readout_rf' / 'readout' / 'res_readout', else a blank "
                "pulse-readout referencing 'ro_waveform' when present). Optional "
                "reset references a library reset ('reset_bath' / 'reset_10' / "
                "'reset_120') when present."
            ),
            typical_writeback=(
                "Proposes the fitted T2-echo time into MetaDict 't2e' (us). No "
                "ModuleLibrary writeback."
            ),
            recommended=(
                "Analysis 'Fit method' defaults to 'decay' (pure exponential "
                "decay → T2E); use 'fringe' instead when a deliberate detuning "
                "leaves a residual oscillation to fit (detune is then computed but "
                "not written back). A total-delay sweep reaching a few T2E lets "
                "the echo decay; the first sweep point is dropped in the fit."
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
                        "pi_pulse": make_pulse_module_spec(),
                        "pi2_pulse": make_pulse_module_spec(),
                        "readout": make_readout_module_spec(),
                    },
                ),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"length": SweepSpec(label="Total delay (us)")},
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t2e", factor=4.0, fallback=20.0)
        relax_delay = proper_relax(ctx)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=relax_delay)
            .role("modules.pi2_pulse", "pi2_pulse")
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
        self, result: T2EchoRunResult, ctx: ExpContext
    ) -> T2EchoAnalyzeParams:
        return T2EchoAnalyzeParams(fit_method="decay")

    def analyze(
        self, req: AnalyzeRequest[T2EchoRunResult, T2EchoAnalyzeParams]
    ) -> T2EchoAnalyzeResult:
        params = req.analyze_params
        t2e, t2e_err, _, _, fig = T2EchoExp().analyze(
            req.run_result,
            fit_method=params.fit_method,
        )
        return T2EchoAnalyzeResult(t2e=t2e, t2e_err=t2e_err, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[T2EchoRunResult, T2EchoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="t2e",
                description="T2 Echo time (us)",
                proposed_value=result.t2e,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2echo_{time.strftime('%m%d')}"
