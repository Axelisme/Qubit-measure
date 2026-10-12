"""jpa/power GUI adapter — JPA pump-power calibration sweep.

Owns the jpa/power cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import custom
from zcu_lab.v2.jpa.power.core import PowerCfg, PowerExp, PowerResult

JpaPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]

_JPA_POWER_SWEEP_EXPTS = 101
_JPA_POWER_SEED_SPAN_DB = 5.0


def jpa_power_sweep_seed(
    ctx: SessionEnv, *, expts: int = _JPA_POWER_SWEEP_EXPTS
) -> SweepValue:
    """Store a ±5 dB sweep around $jpa.pump_power without reading ctx.

    Missing aliases/calibration become Invalid during cfg resolution.
    """
    del ctx  # Defaults retain expressions; the resource owns source reads.
    return SweepValue(
        start=EvalValue(expr=f"$jpa.pump_power - {_JPA_POWER_SEED_SPAN_DB}"),
        stop=EvalValue(expr=f"$jpa.pump_power + {_JPA_POWER_SEED_SPAN_DB}"),
        expts=expts,
    )


@dataclass
class JpaPowerAnalyzeResult(AnalyzeResultBase):
    best_power: float


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
    BaseAdapter[PowerCfg, JpaPowerRunResult, JpaPowerAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
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
            "Uses current aliases: qubit (qubit/*), resonator (resonator), and "
            "jpa (amplifier/jpa). Add a jpa alias row in the global dialog, "
            "for example $resonator.amplifier. Scalars use $jpa.pump_freq, "
            "$jpa.pump_power, $jpa.flux and canonical qubit/resonator wiring. "
            "Missing aliases or calibration leave cfg Invalid; no static fallback."
        ),
        expects_ml=(
            "SMART modules use $qubit.module.x180, $qubit.module.readout and "
            "optional $qubit.module.reset. Inline readout uses resonator scalar "
            "and wiring calibration; module waveforms follow their module source."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing JPA pump power into parameter context "
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
            .relax_delay(0.5)
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
            .reps(10000)
            .rounds(1)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaPowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaPowerRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaPowerAnalyzeResult:
        answer = PowerExp().analyze(req.run_result, None, plots=plots)
        _relabel_power_figure(plots["fit"])
        return JpaPowerAnalyzeResult(best_power=answer.best_power)

    def get_writeback_items(
        self, req: WritebackRequest[JpaPowerRunResult, JpaPowerAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="best_jpa_power",
                description="Best JPA pump power (dBm)",
                proposed_value=req.analyze_result.best_power,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_power_{time.strftime('%m%d')}"
