from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Literal, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_default,
    make_readout_module_spec,
    make_reset_module_spec,
    make_reset_ref_default,
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
    BaseAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, FreqAnalyzeParams]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

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
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        q_f = md_get_float(ctx, "q_f", 4000.0)
        qf_w = md_get_float(ctx, "qf_w", 20.0)
        half_span = 1.5 * qf_w if qf_w > 0 else 30.0
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
        return root_val

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
            md_writeback(ctx, "q_f", "Qubit frequency (MHz)", result.freq),
            md_writeback(ctx, "qf_w", "Qubit linewidth FWHM (MHz)", result.fwhm),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_freq_{time.strftime('%m%d')}"
