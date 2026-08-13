"""jpa/freq GUI adapter — JPA pump-frequency tracer.

Owns the jpa/freq cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_tools.experiment.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.jpa import FreqCfg, FreqExp
from zcu_tools.experiment.v2.jpa.jpa_freq import FreqResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    custom,
    md_has_key,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import EvalValue, SweepValue

from ._shared import cached_device_snapshot, lower_jpa_rf_dev

_JPA_FREQ_SWEEP_EXPTS = 101
# Bring-up seed: ±2% around the centre. These are inspectable starting bounds,
# NOT safety certification — the operator must review device and sweep.
_JPA_FREQ_SEED_SPAN = 0.02
# Fallback centre when neither md key exists: near 2 * r_f with r_f's 6.5 GHz
# bring-up default.
_JPA_FREQ_SEED_CENTER_MHZ = 13000.0


def jpa_freq_sweep_seed(
    ctx: ExpContext, *, expts: int = _JPA_FREQ_SWEEP_EXPTS
) -> SweepValue:
    """JPA pump sweep seed: centred on ``best_jpa_freq`` when known, else near
    ``2 * r_f`` (live expression when ``r_f`` exists, fixed seed otherwise)."""

    if md_has_key(ctx, "best_jpa_freq"):
        center = "best_jpa_freq"
    elif md_has_key(ctx, "r_f"):
        center = "2 * r_f"
    else:
        center_value = _JPA_FREQ_SEED_CENTER_MHZ
        return SweepValue(
            start=center_value * (1 - _JPA_FREQ_SEED_SPAN),
            stop=center_value * (1 + _JPA_FREQ_SEED_SPAN),
            expts=expts,
        )
    return SweepValue(
        start=EvalValue(expr=f"(1 - {_JPA_FREQ_SEED_SPAN}) * ({center})"),
        stop=EvalValue(expr=f"(1 + {_JPA_FREQ_SEED_SPAN}) * ({center})"),
        expts=expts,
    )


@dataclass
class JpaFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    figure: Figure


class JpaFreqAdapter(
    BaseAdapter[FreqCfg, FreqResult, JpaFreqAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA pump-frequency tracer: with the qubit prepared in g and e (a "
            "pi pulse toggles it), sweeps the JPA pump frequency and measures "
            "the g/e signal difference, so you can pick the pump frequency that "
            "best enhances readout. Runs on real hardware. WARNING: review the "
            "selected JPA RF device and the pump sweep before running — the "
            "seeded bounds are bring-up defaults, not certified safety limits, "
            "and the run commands the selected device."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, seeding the pump sweep near 2*r_f; 'best_jpa_freq' — a "
            "previously accepted pump frequency, preferred as the sweep centre; "
            "'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — cable "
            "time-of-flight for the trigger offset; 'q_f' / 'qub_ch' — qubit "
            "frequency / drive channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform named "
            "'ro_waveform' when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing JPA pump frequency into MetaDict "
            "'best_jpa_freq' (MHz) as a draft — it never writes it back without "
            "your acceptance and never touches 'cur_jpa_A'."
        ),
        recommended=(
            "Review the selected RF device and the sweep bounds before every "
            "run; the seeded range near 2*r_f is only a starting point. "
            "Analysis picks the peak of the absolute signal difference."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .device(
                "jpa_rf_dev",
                label="JPA RF device",
                default="",
                required=False,
            )
            .sweep(
                "jpa_freq",
                label="JPA pump freq (MHz)",
                default=custom(
                    jpa_freq_sweep_seed,
                    description="jpa pump frequency range",
                ),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FreqCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_rf_dev(cfg_raw, cached_device_snapshot())
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req
        # Pure preflight over cached/static data — never commands a live device.
        lower_jpa_rf_dev(raw_cfg, cached_device_snapshot())

    def analyze(
        self, req: AnalyzeRequest[FreqResult, NoAnalyzeParams]
    ) -> JpaFreqAnalyzeResult:
        best_freq, fig = FreqExp().analyze(req.run_result)
        return JpaFreqAnalyzeResult(best_freq=best_freq, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[FreqResult, JpaFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            MetaDictWriteback(
                target_name="best_jpa_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=req.analyze_result.best_freq,
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_jpa_freq_{time.strftime('%m%d')}"
