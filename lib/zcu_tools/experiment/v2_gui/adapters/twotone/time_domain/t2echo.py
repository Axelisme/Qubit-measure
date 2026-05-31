from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Literal, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t2echo import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pi2_pulse_ref_default,
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

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t2e", factor=4.0, fallback=20.0)
        relax_delay = proper_relax(ctx)
        _module_fields: dict[str, CfgNodeValue] = {
            "pi2_pulse": make_pi2_pulse_ref_default(ctx),
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
