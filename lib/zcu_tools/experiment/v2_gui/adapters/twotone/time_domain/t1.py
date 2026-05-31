from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pi_pulse_ref_default,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    make_reset_ref_default,
    md_eval_scaled,
    proper_relax,
)
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    MetaDictWriteback,
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
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "pi_pulse": make_pulse_module_spec(),
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
        sweep_stop = md_eval_scaled(ctx, "t1", factor=5.0, fallback=100.0)
        relax_delay = proper_relax(ctx)
        _module_fields: dict[str, CfgNodeValue] = {
            "pi_pulse": make_pi_pulse_ref_default(ctx),
            "readout": make_readout_ref_default(ctx),
        }
        _reset = make_reset_ref_default(ctx, optional=True)
        if _reset is not None:
            _module_fields["reset"] = _reset
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(fields=_module_fields),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": relax_delay,
                "sweep": CfgSectionValue(
                    fields={
                        "length": SweepValue(start=0.0, stop=sweep_stop, expts=101),
                    }
                ),
            }
        )
        return root_val

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
