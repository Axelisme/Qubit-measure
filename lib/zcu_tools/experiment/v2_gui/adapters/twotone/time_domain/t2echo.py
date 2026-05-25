from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Literal, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t2echo import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)
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
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
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
class T2EchoAnalyzeResult:
    t2e: float
    t2e_err: float
    figure: Figure


class T2EchoAdapter(
    AbsExpAdapter[T2EchoRunResult, T2EchoAnalyzeResult, T2EchoAnalyzeParams]
):
    exp_cls = T2EchoExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        t2e = md_get_float(ctx, "t2e", 20.0)
        root_spec = CfgSectionSpec(
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
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"length": SweepSpec(label="Total delay (us)")},
                ),
            }
        )
        _module_fields: dict[str, CfgNodeValue] = {
            "pi2_pulse": make_pulse_ref_default(ctx),
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
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "length": SweepValue(start=0.0, stop=t2e * 4, expts=101),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T2EchoCfg:
        return req.ml.make_cfg(raw_cfg, T2EchoCfg)

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
        ctx = req.ctx
        return [
            MetaDictWriteback(
                key="t2e",
                description="T2 Echo time (us)",
                current_value=ctx.md.get("t2e"),
                md_key="t2e",
                proposed_value=round(result.t2e, 4),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2echo_{time.strftime('%m%d')}"
