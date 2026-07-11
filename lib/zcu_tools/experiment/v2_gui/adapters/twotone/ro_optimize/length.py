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
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    readout_dpm_writeback_items,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    SweepValue,
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("qub_pulse", role_id="pi_pulse")
            .readout(pulse_only=True, locked={"ro_cfg.ro_length": 0.0})
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "length",
                label="Readout length (us)",
                default=SweepValue(start=0.01, stop=3.5, expts=51),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(10000)
            .rounds(1)
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
