"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_lab.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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
from zcu_tools.gui.cfg import EvalValue

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import custom, literal, res_freq_range
from zcu_lab.v2.jpa.flux.gui import jpa_flux_sweep_seed
from zcu_lab.v2.jpa.flux_onetone.core import (
    OneToneFluxCfg,
    OneToneFluxExp,
    OneToneFluxResult,
)

JpaFluxOneToneRunResult: TypeAlias = RunRecord[OneToneFluxCfg, OneToneFluxResult]

# Bring-up survey: ~101 readout-frequency points around the resonator; the
# flux sweep reuses the shared JPA flux survey seed (101 points). These are
# inspectable starting bounds, NOT safety certification — the operator must
# review device and sweeps.
_JPA_FLUX_ONETONE_FREQ_EXPTS = 101


class JpaFluxOneToneAdapter(BaseAdapter[OneToneFluxCfg, JpaFluxOneToneRunResult]):
    exp_cls = OneToneFluxExp
    ExpCfg_cls: ClassVar[Any] = OneToneFluxCfg
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
            "No writeback — this adapter has no analysis step (the "
            "underlying experiment has no optimum analysis yet). It produces "
            "a 2D map for visual inspection only; read off useful flux / "
            "frequency settings by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis options. Typical sweep: JPA flux device value ~101 "
            "points around the seeded centre ($jpa.flux), "
            "readout frequency ~101 points spanning 1.5 linewidths around "
            "'r_f'. Confirm the JPA pump state yourself before every run. "
            "Survey the map wide first, then narrow the ranges around the "
            "response you want to inspect."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .choice(
                "flux_unit",
                label="Flux unit",
                choices=("A", "V", "native"),
                default="A",
                tooltip="Working unit for the selected flux driver; no conversion.",
            )
            .reset(optional=True)
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                },
            )
            .relax_delay(0.1)
            .device(
                "jpa_flux_dev",
                label="JPA flux device",
                default=literal(EvalValue("$flux_dev")),
                required=False,
            )
            .sweep(
                "jpa_flux",
                label="JPA flux device value",
                default=custom(
                    jpa_flux_sweep_seed,
                    description="jpa flux device value range",
                ),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=_JPA_FLUX_ONETONE_FREQ_EXPTS),
            )
            .reps(100)
            .rounds(10)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaFluxOneToneRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, OneToneFluxExp().run(cfg, context=context))

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_flux_onetone_{time.strftime('%m%d')}"
