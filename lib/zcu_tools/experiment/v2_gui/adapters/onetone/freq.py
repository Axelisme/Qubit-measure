from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment.v2.onetone.freq import FreqCfg, FreqExp
from zcu_tools.experiment.v2_gui.adapters.shared import (
    build_readout_for_frequency,
    build_waveform_for_length,
    make_flat_top_waveform_edit_template,
    make_module_ref_default,
    make_pulse_readout_ref_spec,
    make_readout_edit_template,
    make_reset_ref_spec,
    require_soc_handles,
    save_with_last_state,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ModuleWriteback,
    ParamMeta,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.specs.readout import make_pulse_readout_spec
from zcu_tools.program.v2 import PulseReadoutCfg


@dataclass
class OneToneFreqRunResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqCfg


@dataclass
class OneToneFreqAnalyzeParams:
    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")]
    fit_bg_slope: Annotated[bool, ParamMeta(label="Fit background slope")]


@dataclass
class OneToneFreqAnalyzeResult:
    freq: float
    fwhm: float
    params: dict[str, Any]
    figure: Figure


def _md_float(ctx: ExpContext, key: str, default: float) -> float:
    value = getattr(ctx.md, key, None)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _pulse_readout_default(ctx: ExpContext):
    return make_module_ref_default(
        ml=ctx.ml,
        module_type=PulseReadoutCfg,
        preferred_names=["readout_rf", "readout", "res_readout"],
        fallback_key="<Custom:Pulse Readout>",
        fallback_spec_factory=make_pulse_readout_spec,
    )


class OneToneFreqAdapter(
    AbsExpAdapter[
        OneToneFreqRunResult, OneToneFreqAnalyzeResult, OneToneFreqAnalyzeParams
    ]
):
    exp_cls = FreqExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        r_f = _md_float(ctx, "r_f", 6000.0)
        rf_w = _md_float(ctx, "rf_w", 20.0)
        half_span = 1.5 * rf_w if rf_w > 0 else 30.0
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "readout": make_pulse_readout_ref_spec(),
                        "reset": make_reset_ref_spec(optional=True),
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
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": _pulse_readout_default(ctx),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "freq": SweepValue(
                            start=r_f - half_span,
                            stop=r_f + half_span,
                            expts=301,
                        )
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FreqCfg:
        return req.ml.make_cfg(raw_cfg, FreqCfg)

    def run(self, req: RunRequest, schema: CfgSchema) -> OneToneFreqRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        freqs, signals = FreqExp().run(soc, soccfg, cfg)
        return OneToneFreqRunResult(freqs=freqs, signals=signals, cfg_snapshot=cfg)

    def get_analyze_params(
        self, result: OneToneFreqRunResult, ctx: ExpContext
    ) -> OneToneFreqAnalyzeParams:
        return OneToneFreqAnalyzeParams(model_type="hm", fit_bg_slope=True)

    def analyze(
        self,
        req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams],
    ) -> OneToneFreqAnalyzeResult:
        params = req.analyze_params
        result = req.run_result
        freq, fwhm, fit_params, figure = FreqExp().analyze(
            (result.freqs, result.signals),
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
        ctx = req.ctx
        readout = req.run_result.cfg_snapshot.modules.readout
        pulse_ch = getattr(ctx.md, "res_ch", 0)
        ro_ch = getattr(ctx.md, "ro_ch", 0)
        wav_len = _md_float(ctx, "res_probe_len", 5.0)

        return [
            MetaDictWriteback(
                key="r_f",
                description="Resonator frequency (MHz)",
                current_value=getattr(ctx.md, "r_f", None),
                md_key="r_f",
                proposed_value=round(result.freq, 4),
            ),
            MetaDictWriteback(
                key="rf_w",
                description="Resonator linewidth FWHM (MHz)",
                current_value=getattr(ctx.md, "rf_w", None),
                md_key="rf_w",
                proposed_value=round(result.fwhm, 4),
            ),
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

    def save(self, req: SaveDataRequest[OneToneFreqRunResult]) -> None:
        result = req.run_result
        save_with_last_state(
            exp_cls=FreqExp,
            cfg=result.cfg_snapshot,
            result=(result.freqs, result.signals),
            filepath=req.data_path,
        )
