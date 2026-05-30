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
    build_readout_for_frequency,
    build_waveform_for_length,
    make_flat_top_waveform_edit_template,
    make_pulse_readout_default,
    make_pulse_readout_module_spec,
    make_readout_edit_template,
    make_reset_module_spec,
    md_get_float,
    md_has_key,
    md_writeback,
)
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ExpContext,
    ModuleWriteback,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformWriteback,
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
        spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "readout": make_pulse_readout_module_spec(),
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
        # The readout pulse/ro frequency is driven by the sweep axis, not the
        # user (notebook: ``freq: 0.0, # not used``). Lock both to 0.0 — the lock
        # is part of the spec contract, hence returned from cfg_spec().
        return spec.lock_literal(
            "modules.readout.pulse_cfg.freq", 0.0
        ).lock_literal("modules.readout.ro_cfg.ro_freq", 0.0)

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        r_f = md_get_float(ctx, "r_f", 6000.0)
        rf_w = md_get_float(ctx, "rf_w", 20.0)
        half_span = 1.5 * rf_w if rf_w > 0 else 30.0
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: Union[float, EvalValue] = (
            EvalValue(
                expr="res_probe_len - 0.1",
                resolved=probe_len - 0.1,
                error=None,
            )
            if md_has_key(ctx, "res_probe_len")
            else probe_len - 0.1
        )
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_pulse_readout_default(
                            ctx, gain=0.05, ro_length=ro_length
                        ),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "freq": SweepValue(
                            start=(
                                EvalValue(
                                    expr="r_f - 1.5 * rf_w",
                                    resolved=r_f - half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f - half_span)
                            ),
                            stop=(
                                EvalValue(
                                    expr="r_f + 1.5 * rf_w",
                                    resolved=r_f + half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f + half_span)
                            ),
                            expts=301,
                        )
                    }
                ),
            }
        )
        return root_val

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

        ctx = req.ctx
        readout = cfg.modules.readout
        pulse_ch = ctx.md.get("res_ch", 0)
        ro_ch = ctx.md.get("ro_ch", 0)
        wav_len = md_get_float(ctx, "res_probe_len", 5.0)

        return [
            md_writeback(ctx, "r_f", "Resonator frequency (MHz)", result.freq),
            md_writeback(ctx, "rf_w", "Resonator linewidth FWHM (MHz)", result.fwhm),
            ModuleWriteback(
                key="readout_rf",
                description="readout_rf module config",
                current_value=ctx.ml.modules.get("readout_rf"),
                module_name="readout_rf",
                proposed_module=build_readout_for_frequency(
                    readout,
                    freq=result.freq,
                    pulse_ch=pulse_ch,
                    ro_ch=ro_ch,
                    ml=ctx.ml,
                ),
                edit_schema=make_readout_edit_template(
                    readout,
                    freq=result.freq,
                    pulse_ch=pulse_ch,
                    ro_ch=ro_ch,
                ),
            ),
            WaveformWriteback(
                key="ro_waveform",
                description="ro_waveform length config",
                current_value=ctx.ml.waveforms.get("ro_waveform"),
                waveform_name="ro_waveform",
                proposed_waveform=build_waveform_for_length(
                    readout,
                    length=wav_len,
                    ml=ctx.ml,
                ),
                edit_schema=make_flat_top_waveform_edit_template(length=wav_len),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_freq_{time.strftime('%m%d')}"
