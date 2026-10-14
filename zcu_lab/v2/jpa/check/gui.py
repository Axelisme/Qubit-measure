"""jpa/check GUI adapter — JPA pump off/on resonator comparison check.

Owns the jpa/check cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``
(``CheckExp``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
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
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.check.core import CheckCfg, CheckExp, CheckResult
from zcu_lab.v2.jpa.check.form import FORM

JpaCheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


@dataclass
class JpaCheckAnalyzeResult(AnalyzeResultBase):
    pass


class JpaCheckAdapter(
    BaseAdapter[CheckCfg, JpaCheckRunResult, JpaCheckAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = CheckExp.persistence_tag()
    exp_cls = CheckExp
    ExpCfg_cls: ClassVar[Any] = CheckCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA pump off/on resonator check: sweeps the readout frequency "
            "with the JPA pump RF output off and again with it on, recording "
            "the one-tone resonator response in both states so you can "
            "compare the resonance by eye. Runs on real hardware and commands "
            "the selected JPA RF device (its output is toggled off, then on). "
            "WARNING: the run leaves the JPA pump output ON when it finishes "
            "— the pump is NOT turned off afterwards. Review the selected RF "
            "device and the sweep range before running — the seeded bounds "
            "are bring-up defaults, not certified safety limits."
        ),
        expects_md=(
            "Uses the resonator alias for $resonator.freq, $resonator.kappa and "
            "resonator wiring. The qubit alias supplies an optional reset module. "
            "Select the connected JPA RF device in the form. RF defaults capture "
            "jpa_rf_dev at Open or Reset; an unbound alias leaves the optional "
            "field empty. This check does not use "
            "jpa pump-frequency, pump-power or flux calibration. Missing required "
            "resonator calibration leaves the form Invalid."
        ),
        expects_ml=(
            "Readout starts as a fresh readout template, not a library copy. Its "
            "inline pulse uses $resonator.freq and $resonator.wiring.ch; acquisition "
            "uses $resonator.freq, $resonator.wiring.ro_ch and "
            "$resonator.ext.time_of_fly + 0.05. Run replaces both frequencies from the "
            "sweep. Readout library keys remain selectable. Optional reset uses "
            "$qubit.module.reset; an unset slot disables reset."
        ),
        typical_writeback=(
            "No writeback — the check is a visual diagnostic: compare the two "
            "traces and update parameters in another step."
        ),
        recommended=(
            "No analysis options — the Analyze tab shows the pump off/on "
            "comparison figure only. Typical sweep: readout frequency ~101 "
            "points spanning 1.5 linewidths around $resonator.freq. Before every run, "
            "confirm the current JPA pump state and review the selected RF "
            "device: the run toggles its output off then on and leaves it ON."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, CheckExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaCheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaCheckAnalyzeResult:
        CheckExp().analyze(req.run_result, None, plots=plots)
        return JpaCheckAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_check_{time.strftime('%m%d')}"
