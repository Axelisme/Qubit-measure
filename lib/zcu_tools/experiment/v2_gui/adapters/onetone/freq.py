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
    Union,
)

from zcu_tools.experiment.v2.onetone.freq import FreqCfg, FreqExp, FreqResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_onetone_freq_writeback_items,
    make_pulse_readout_default,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    md_get_float,
    md_has_key,
    proper_res_freq_range,
)
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ExpContext,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

OneToneFreqRunResult: TypeAlias = FreqResult


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")]
    fit_bg_slope: Annotated[bool, ParamMeta(label="Fit background slope")]


@dataclass
class OneToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


class OneToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        OneToneFreqRunResult,
        OneToneFreqAnalyzeResult,
        OneToneFreqAnalyzeParams,
    ]
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
                        "readout": make_pulse_readout_module_spec()
                        .lock_literal("pulse_cfg.freq", 0.0)
                        .lock_literal("ro_cfg.ro_freq", 0.0),
                    },
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"freq": SweepSpec(label="Freq (MHz)")},
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: Union[float, EvalValue] = (
            EvalValue(expr="res_probe_len - 0.1")
            if md_has_key(ctx, "res_probe_len")
            else probe_len - 0.1
        )
        return CfgSectionValue(
            {
                "modules": CfgSectionValue(
                    {
                        "readout": make_pulse_readout_default(ctx)
                        .with_field("pulse_cfg.gain", 0.05)
                        .with_field("ro_cfg.ro_length", ro_length),
                    }
                ),
                "sweep": CfgSectionValue(
                    fields={"freq": proper_res_freq_range(ctx, 301)},
                ),
                "relax_delay": DirectValue(1.0),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
            }
        )

    def get_analyze_params(
        self, result: OneToneFreqRunResult, ctx: ExpContext
    ) -> OneToneFreqAnalyzeParams:
        return OneToneFreqAnalyzeParams(model_type="hm", fit_bg_slope=True)

    def analyze(
        self, req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams]
    ) -> OneToneFreqAnalyzeResult:
        params = req.analyze_params
        freq, fwhm, fit_params, figure = FreqExp().analyze(
            req.run_result,
            model_type=params.model_type,
            fit_bg_slope=params.fit_bg_slope,
        )
        return OneToneFreqAnalyzeResult(
            freq=freq,
            fwhm=fwhm,
            params=fit_params,
            figure=figure,
        )

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFreqRunResult, OneToneFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        cfg = req.run_result.cfg_snapshot
        assert cfg is not None, "cfg_snapshot is required for writeback"
        return make_onetone_freq_writeback_items(
            cfg.modules.readout, result.freq, result.fwhm, req.ctx
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"
