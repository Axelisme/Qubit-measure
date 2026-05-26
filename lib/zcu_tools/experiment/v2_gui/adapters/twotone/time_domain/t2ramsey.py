from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    make_reset_ref_default,
    md_get_float,
    md_has_key,
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

T2RamseyRunResult: TypeAlias = T2RamseyResult


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")]


@dataclass
class T2RamseyAnalyzeResult:
    t2r: float
    t2r_err: float
    detune: float
    figure: Figure


class T2RamseyAdapter(
    AbsExpAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    exp_cls = T2RamseyExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        t1 = md_get_float(ctx, "t1", 100.0)
        t2r = md_get_float(ctx, "t2r", 20.0)
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
        _module_fields: dict[str, CfgNodeValue] = {
            "pi2_pulse": make_pulse_ref_default(
                ctx, preferred_names=["pi2_amp", "pi2_len", "pi_amp", "pi_len"]
            ),
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
                        "length": SweepValue(start=0.0, stop=t2r * 4, expts=101),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T2RamseyCfg:
        return req.ml.make_cfg(raw_cfg, T2RamseyCfg)

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
        ctx = req.ctx
        return [
            MetaDictWriteback(
                key="t2r",
                description="T2 Ramsey time (us)",
                current_value=ctx.md.get("t2r"),
                md_key="t2r",
                proposed_value=round(result.t2r, 4),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2ramsey_{time.strftime('%m%d')}"
