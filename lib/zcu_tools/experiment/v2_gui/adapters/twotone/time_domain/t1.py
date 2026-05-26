from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    make_reset_ref_default,
    md_get_float,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    ScalarSpec,
    ScalarValue,
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
class T1AnalyzeResult:
    t1: float
    t1_err: float
    figure: Figure


class T1Adapter(AbsExpAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    exp_cls = T1Exp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        from zcu_tools.experiment.v2_gui.adapters.shared import md_has_key

        t1 = md_get_float(ctx, "t1", 100.0)
        relax_delay: ScalarValue = (
            EvalValue(expr="5 * t1", resolved=5.0 * t1, error=None)
            if md_has_key(ctx, "t1")
            else DirectValue(100.0)
        )
        root_spec = CfgSectionSpec(
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
        _module_fields: dict[str, CfgNodeValue] = {
            "pi_pulse": make_pulse_ref_default(ctx),
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
                        "length": SweepValue(start=0.0, stop=t1 * 5, expts=101),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T1Cfg:
        return req.ml.make_cfg(raw_cfg, T1Cfg)

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
        ctx = req.ctx
        return [
            MetaDictWriteback(
                key="t1",
                description="T1 relaxation time (us)",
                current_value=ctx.md.get("t1"),
                md_key="t1",
                proposed_value=round(result.t1, 4),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
