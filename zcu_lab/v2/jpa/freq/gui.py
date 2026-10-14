"""jpa/freq GUI adapter — JPA pump-frequency tracer.

Owns the jpa/freq cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
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
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import custom
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.jpa.freq.core import FreqCfg, FreqExp, FreqResult

JpaFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]

_JPA_FREQ_SWEEP_EXPTS = 101
# Bring-up seed: ±2% around the centre. These are inspectable starting bounds,
# NOT safety certification — the operator must review device and sweep.
_JPA_FREQ_SEED_SPAN = 0.02


def jpa_freq_sweep_seed(
    ctx: SessionEnv, *, expts: int = _JPA_FREQ_SWEEP_EXPTS
) -> SweepValue:
    """Store ±2% around $jpa.pump_freq without reading ctx calibration.

    Missing aliases/calibration become Invalid during cfg resolution.
    """
    del ctx  # Defaults retain expressions; the resource owns source reads.
    return SweepValue(
        start=EvalValue(expr=f"(1 - {_JPA_FREQ_SEED_SPAN}) * $jpa.pump_freq"),
        stop=EvalValue(expr=f"(1 + {_JPA_FREQ_SEED_SPAN}) * $jpa.pump_freq"),
        expts=expts,
    )


@dataclass
class JpaFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class JpaFreqAdapter(
    BaseAdapter[FreqCfg, JpaFreqRunResult, JpaFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
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
                "jpa_freq",
                label="JPA pump freq (MHz)",
                default=custom(
                    jpa_freq_sweep_seed,
                    description="jpa pump frequency range",
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
                target_name="best_jpa_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=req.analyze_result.best_freq,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_freq_{time.strftime('%m%d')}"
