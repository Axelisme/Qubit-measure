from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.ro_optimize.length import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    proper_relax,
    readout_dpm_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    FloatSpec,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

RoOptLengthRunResult: TypeAlias = LengthResult


@dataclass
class RoOptLengthAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    # GUI-facing name for LengthExp.analyze(t0=...): this is a duration
    # normalization overhead, not a penalty strength. Small positive values
    # produce stronger short-readout bias than large positive values.
    duration_t0: Annotated[float | None, ParamMeta(label="Duration t0 (us)")] = None


@dataclass
class RoOptLengthAnalyzeResult(AnalyzeResultBase):
    best_length: float
    figure: Figure


class RoOptLengthAdapter(
    BaseAdapter[
        LengthCfg,
        RoOptLengthRunResult,
        RoOptLengthAnalyzeResult,
        RoOptLengthAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout length optimization: with the qubit toggled between g and "
            "e by a pi pulse, sweeps the readout window length and measures the "
            "g/e signal-to-noise ratio (SNR), to pick the shortest readout that "
            "still resolves the states well. Runs on real hardware. A "
            "readout-tuning step, typically after the readout frequency and "
            "power are set."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
            "resonator / chosen readout frequency (~4000–8000 MHz); 'res_ch' / "
            "'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable "
            "delay; 'q_f' / 'qub_ch' — qubit frequency / channel for the g↔e "
            "pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf', usually pinned to the chosen readout frequency and "
            "gain); references a ModuleLibrary waveform 'ro_waveform' when "
            "present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout length into MetaDict "
            "'best_ro_length' (us). When a cfg snapshot with pulse readout is "
            "available and 'best_ro_freq' / 'best_ro_gain' / "
            "'best_ro_length' are known from this result plus MetaDict, also "
            "proposes ModuleLibrary 'readout_dpm'."
        ),
        recommended=(
            "Analysis denoises the SNR curve before picking the peak; wavelet "
            "smoothing is the default. The optional 'duration_t0' analyze param "
            "is the fixed-duration term in SNR/sqrt(length + t0): leave it "
            "blank (None) for the raw max; a small positive us value applies "
            "stronger short-readout bias than a large positive value."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                # The sweep axis owns the readout acquisition window
                # (set_param("ro_length") at run), so lock it off the
                # form. The pulse waveform length is auto-derived from the
                # sweep max but stays editable because shaped pulses use it to
                # set their Gaussian ratio.
                "readout": make_pulse_readout_module_spec().lock_literal(
                    "ro_cfg.ro_length", 0.0
                ),
            },
            sweep={"length": SweepSpec(label="Readout length (us)")},
            extra={"skew_penalty": FloatSpec(label="Skew penalty", decimals=3)},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=10000,
                rounds=1,
                relax_delay=proper_relax(ctx),
                skew_penalty=0.0,
            )
            .role("modules.reset", "reset", Init.DISABLED)
            .role("modules.qub_pulse", "pi_pulse")
            .role("modules.readout", "readout")
            .sweep("sweep.length", 0.01, 3.5, 51)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[RoOptLengthRunResult, RoOptLengthAnalyzeParams]
    ) -> RoOptLengthAnalyzeResult:
        params = req.analyze_params
        best_length, fig = LengthExp().analyze(
            req.run_result,
            t0=params.duration_t0,
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        return RoOptLengthAnalyzeResult(best_length=best_length, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptLengthRunResult, RoOptLengthAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                proposed={"best_ro_length": result.best_length},
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_length_{time.strftime('%m%d')}"
