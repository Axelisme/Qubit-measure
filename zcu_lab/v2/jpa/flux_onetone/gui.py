"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_lab.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

import time
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.flux_onetone.core import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxResult,
)
from zcu_lab.v2.jpa.flux_onetone.form import make_form

JpaFluxOneToneRunResult: TypeAlias = RunRecord[OneToneFluxCfg, OneToneFluxResult]


class JpaFluxOneToneAdapter(BaseAdapter[OneToneFluxCfg, JpaFluxOneToneRunResult]):
    experiment_tag: ClassVar[str | None] = OneToneFluxExp.persistence_tag()
    exp_cls = OneToneFluxExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[OneToneFluxCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA flux × readout-frequency 2D survey: sweeps the JPA flux "
            "device value (outer axis) against the readout frequency (inner "
            "axis) and records the one-tone resonator response, so you can "
            "see by eye how the JPA response changes with flux and frequency. "
            "There is no automatic optimum analysis — judge the 2D map "
            "directly. Runs on real hardware and commands the selected JPA "
            "flux device. WARNING: review the selected flux device and both "
            "sweep ranges before running — the seeded bounds are bring-up "
            "defaults, not certified safety limits. The JPA pump is NOT "
            "controlled by this experiment: if the pump must be on for the "
            "measurement, confirm its state yourself (in the Devices "
            "interface) before running."
        ),
        expects_md=(
            "Uses the resonator and jpa (amplifier/jpa) aliases. Bounds use "
            "$resonator.freq, $resonator.kappa and $jpa.flux. The required JPA flux "
            "device resolves flux_dev once at Open or Reset; an unbound alias leaves "
            "the form Invalid. flux_unit starts as A with no conversion. The qubit "
            "alias supplies an optional reset module."
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
            "No writeback — this adapter has no analysis step (the "
            "underlying experiment has no optimum analysis yet). It produces "
            "a 2D map for visual inspection only; read off useful flux / "
            "frequency settings by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis options. Typical sweep: JPA flux device value ~101 "
            "points around the seeded centre ($jpa.flux), "
            "readout frequency ~101 points spanning 1.5 linewidths around "
            "$resonator.freq. Confirm the JPA pump state before every run. "
            "Survey the map wide first, then narrow the ranges around the "
            "response you want to inspect."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: OneToneFluxCfg,
        *,
        context: RunContext,
    ) -> JpaFluxOneToneRunResult:
        del req
        return RunRecord(cfg, OneToneFluxExp().run(cfg, context=context))

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_flux_onetone_{time.strftime('%m%d')}"
