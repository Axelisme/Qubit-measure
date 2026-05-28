from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, Literal, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
    md_get_float,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    AnalyzeResultBase,
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

FreqRunResult: TypeAlias = FreqResult


@dataclass
class FreqAnalyzeParams:
    model_type: Annotated[Literal["lor", "sinc"], ParamMeta(label="Model type")]
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")]


@dataclass
class FreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


class FreqAdapter(
    AbsExpAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, FreqAnalyzeParams]
):
    exp_cls = FreqExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        q_f = md_get_float(ctx, "q_f", 4000.0)
        qf_w = md_get_float(ctx, "qf_w", 20.0)
        half_span = 1.5 * qf_w if qf_w > 0 else 30.0
        root_spec = CfgSectionSpec(
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
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
                ),
            }
        )
        _module_fields: dict[str, CfgNodeValue] = {
            "qub_pulse": make_pulse_ref_default(ctx),
            "readout": make_readout_default(ctx),
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
                        "freq": SweepValue(
                            start=q_f - half_span,
                            stop=q_f + half_span,
                            expts=301,
                        )
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FreqCfg:
        return req.ml.make_cfg(raw_cfg, FreqCfg)

    def get_analyze_params(
        self, result: FreqRunResult, ctx: ExpContext
    ) -> FreqAnalyzeParams:
        return FreqAnalyzeParams(model_type="lor", plot_fit=True)

    def analyze(
        self, req: AnalyzeRequest[FreqRunResult, FreqAnalyzeParams]
    ) -> FreqAnalyzeResult:
        params = req.analyze_params
        freq, fwhm, fig = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            plot_fit=params.plot_fit,
        )
        return FreqAnalyzeResult(freq=freq, fwhm=fwhm, params={}, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[FreqRunResult, FreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        ctx = req.ctx
        return [
            MetaDictWriteback(
                key="q_f",
                description="Qubit frequency (MHz)",
                current_value=ctx.md.get("q_f"),
                md_key="q_f",
                proposed_value=round(result.freq, 4),
            ),
            MetaDictWriteback(
                key="qf_w",
                description="Qubit linewidth FWHM (MHz)",
                current_value=ctx.md.get("qf_w"),
                md_key="qf_w",
                proposed_value=round(result.fwhm, 4),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_freq_{time.strftime('%m%d')}"
