"""jpa/freq GUI adapter — JPA pump-frequency tracer.

Owns the jpa/freq cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.jpa.freq.form import FORM

JpaFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class JpaFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class JpaFreqAdapter(
    BaseAdapter[FreqCfg, JpaFreqRunResult, JpaFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg
    form = FORM

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
            "Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. The "
            "pump-frequency bounds use $jpa.pump_freq. Select the connected JPA RF "
            "device in the form. RF defaults capture jpa_rf_dev at Open or Reset; "
            "an unbound alias leaves the optional field empty. Missing required "
            "aliases or calibration leave the "
            "form Invalid."
        ),
        expects_ml=(
            "Module slots use $qubit.module.x180, $qubit.module.readout and optional "
            "$qubit.module.reset. Missing pi/readout slots fall back to the "
            "pi_pulse/readout templates. Inline pulse calibration uses $qubit.freq and "
            "$qubit.wiring.L01.ch; inline readout uses resonator frequency/wiring and "
            "$resonator.ext.time_of_fly + 0.05. An unset reset slot disables reset. "
            "Missing required fallback calibration leaves the form Invalid."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing JPA pump frequency into "
            "$jpa.pump_freq (MHz) as a draft. Nothing is written without your "
            "acceptance, and writeback never changes the current-source value."
        ),
        recommended=(
            "Review the selected RF device and the sweep bounds before every "
            "run; the seeded range around $jpa.pump_freq is only a starting point. "
            "Analysis picks the peak of the absolute signal difference."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaFreqRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaFreqAnalyzeResult:
        answer = FreqExp().analyze(req.run_result, None, plots=plots)
        return JpaFreqAnalyzeResult(best_freq=answer.best_freq)

    def get_writeback_items(
        self, req: WritebackRequest[JpaFreqRunResult, JpaFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="$jpa.pump_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=req.analyze_result.best_freq,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_freq_{time.strftime('%m%d')}"
