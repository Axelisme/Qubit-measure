"""jpa/power GUI adapter — JPA pump-power calibration sweep.

Owns the jpa/power cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_tools.experiment.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.jpa import PowerCfg, PowerExp
from zcu_tools.experiment.v2.jpa.jpa_power import PowerResult
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

from ._shared import cached_device_snapshot, lower_jpa_rf_power_dev

_JPA_POWER_SWEEP_EXPTS = 101
# Conservative low-power survey: the low-power portion (-20..-5 dBm) of the
# notebook's JPA power sweep (single_qubit.md, -20..1 dBm), compressed to 101
# points. These are inspectable starting bounds, NOT safety certification —
# the operator must review device and sweep.
_JPA_POWER_SEED_START_DBM = -20.0
_JPA_POWER_SEED_STOP_DBM = -5.0
# Refinement span around a previously accepted best_jpa_power (dBm).
_JPA_POWER_SEED_SPAN_DB = 5.0


def jpa_power_sweep_seed(
    ctx: ExpContext, *, expts: int = _JPA_POWER_SWEEP_EXPTS
) -> SweepValue:
    """JPA pump sweep seed: centred on ``best_jpa_power`` when known, else a
    conservative low-power notebook-derived survey (-20..-5 dBm)."""

    if md_has_key(ctx, "best_jpa_power"):
        return SweepValue(
            start=EvalValue(expr=f"best_jpa_power - {_JPA_POWER_SEED_SPAN_DB}"),
            stop=EvalValue(expr=f"best_jpa_power + {_JPA_POWER_SEED_SPAN_DB}"),
            expts=expts,
        )
    return SweepValue(
        start=_JPA_POWER_SEED_START_DBM,
        stop=_JPA_POWER_SEED_STOP_DBM,
        expts=expts,
    )


@dataclass
class JpaPowerAnalyzeResult(AnalyzeResultBase):
    best_power: float
    figure: Figure


def _relabel_power_figure(fig: Figure) -> None:
    """Correct the GUI review figure's x-axis to the swept quantity.

    The core analysis figure mislabels its x-axis as 'JPA Frequency (MHz)'
    although the swept data is pump power in dBm; the GUI review figure must
    say 'JPA pump power (dBm)'. The relabel happens at the adapter analysis
    boundary, in place, leaving the core figure and the Result/writeback
    values untouched.
    """
    for ax in fig.axes:
        ax.set_xlabel("JPA pump power (dBm)")


class JpaPowerAdapter(
    BaseAdapter[PowerCfg, PowerResult, JpaPowerAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA pump-power calibration: with the qubit prepared in g and e "
            "(a pi pulse toggles it), sweeps the JPA pump power (dBm) and "
            "measures the g/e signal difference, so you can pick the pump "
            "power that best enhances readout. Runs on real hardware. WARNING: "
            "review the selected JPA RF device and the power sweep before "
            "running — the seeded bounds are bring-up defaults, not certified "
            "safety limits, and the run commands the selected device."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'best_jpa_power' — a "
            "previously accepted JPA pump power, preferred as the sweep "
            "centre; 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — "
            "cable time-of-flight for the trigger offset; 'q_f' / 'qub_ch' — "
            "qubit frequency / drive channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform named "
            "'ro_waveform' when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing JPA pump power into MetaDict "
            "'best_jpa_power' (dBm) as a draft — it never writes it back "
            "without your acceptance, never commands a device from the "
            "writeback, and never touches 'cur_jpa_A'."
        ),
        recommended=(
            "Review the selected RF device and the power sweep bounds before "
            "every run; the seeded range is a conservative low-power survey, "
            "only a starting point. Analysis picks the peak of the absolute "
            "signal difference."
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
                "jpa_power",
                label="JPA pump power (dBm)",
                default=custom(
                    jpa_power_sweep_seed,
                    description="jpa pump power range",
                ),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> PowerCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_rf_power_dev(cfg_raw, cached_device_snapshot())
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req
        # Pure preflight over cached/static data — never commands a live device.
        lower_jpa_rf_power_dev(raw_cfg, cached_device_snapshot())

    def analyze(
        self, req: AnalyzeRequest[PowerResult, NoAnalyzeParams]
    ) -> JpaPowerAnalyzeResult:
        best_power, fig = PowerExp().analyze(req.run_result)
        _relabel_power_figure(fig)
        return JpaPowerAnalyzeResult(best_power=best_power, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[PowerResult, JpaPowerAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            MetaDictWriteback(
                target_name="best_jpa_power",
                description="Best JPA pump power (dBm)",
                proposed_value=req.analyze_result.best_power,
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_jpa_power_{time.strftime('%m%d')}"
