from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.rabi.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    default_qub_probe,
    default_reset,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    md_get_float,
    md_writeback,
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
    WritebackItem,
    WritebackRequest,
)

LenRabiRunResult: TypeAlias = LenRabiResult


@dataclass
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")]


@dataclass
class LenRabiAnalyzeResult(AnalyzeResultBase):
    pi_len: float
    pi2_len: float
    figure: Figure


class LenRabiAdapter(
    BaseAdapter[
        LenRabiCfg,
        LenRabiRunResult,
        LenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "qub_pulse": make_pulse_module_spec(),
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
                    fields={"length": SweepSpec(label="Length (us)")},
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        pi_len = md_get_float(ctx, "pi_len", 0.1)
        _module_fields: dict[str, CfgNodeValue] = {
            "qub_pulse": default_qub_probe(ctx),
            "readout": make_readout_ref_default(ctx),
        }
        _reset = default_reset(ctx, optional=True)
        if _reset is not None:
            _module_fields["reset"] = _reset
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(fields=_module_fields),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(10.5),
                "sweep": CfgSectionValue(
                    fields={
                        "length": SweepValue(start=0.03, stop=pi_len * 4, expts=101),
                    }
                ),
            }
        )
        return root_val

    def get_analyze_params(
        self, result: LenRabiRunResult, ctx: ExpContext
    ) -> LenRabiAnalyzeParams:
        return LenRabiAnalyzeParams(decay=True)

    def analyze(
        self, req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeParams]
    ) -> LenRabiAnalyzeResult:
        params = req.analyze_params
        pi_len, pi2_len, _, fig = LenRabiExp().analyze(
            req.run_result,
            decay=params.decay,
        )
        return LenRabiAnalyzeResult(pi_len=pi_len, pi2_len=pi2_len, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[LenRabiRunResult, LenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        ctx = req.ctx
        return [
            md_writeback(ctx, "pi_len", "Pi pulse length (us)", result.pi_len, 5),
            md_writeback(ctx, "pi2_len", "Pi/2 pulse length (us)", result.pi2_len, 5),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_len_rabi_{time.strftime('%m%d')}"
