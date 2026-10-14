"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_lab.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_tools.gui.app.measure.seeds import alias_name_seed, custom

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import res_freq_range
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
    experiment_tag: ClassVar[str | None] = OneToneFluxExp.persistence_tag()
    exp_cls = OneToneFluxExp
    ExpCfg_cls: ClassVar[Any] = OneToneFluxCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
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
                default=alias_name_seed("flux_dev"),
                required=False,
            )
            .choice(
                "flux_unit",
                label="Flux unit",
                choices=("A", "V", "native"),
                default="A",
                tooltip="Working unit for the selected flux driver; no conversion.",
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
