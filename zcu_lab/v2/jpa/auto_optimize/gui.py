"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from matplotlib.figure import Figure
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
from zcu_tools.gui.app.measure.seeds import alias_name_seed, custom
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
)
from zcu_lab.v2.jpa.flux.gui import jpa_flux_sweep_seed
from zcu_lab.v2.jpa.freq.gui import jpa_freq_sweep_seed
from zcu_lab.v2.jpa.power.gui import jpa_power_sweep_seed

JpaAutoRunResult = RunRecord[JPAOptCfg, JPAOptimizeResult]

# Bring-up iteration budget (contract: GUI default may use 1001).
_JPA_AUTO_NUM_POINTS_DEFAULT = 1001


def _relabel_auto_figure(fig: Figure) -> None:
    """Neutral 'JPA flux device value' wording on the GUI review figure.

    The core auto-optimize analysis figure labels its flux subplot 'JPA Flux
    value (a.u.)'; the GUI review figure must use the neutral vocabulary of
    the selected flux device without claiming a physical-unit migration. Only
    the flux axis is relabelled — the frequency and power axes keep their own
    quantities.
    """
    for ax in fig.axes:
        if "Flux" in ax.get_xlabel():
            ax.set_xlabel("JPA flux device value")


@dataclass
class JpaAutoAnalyzeResult(AnalyzeResultBase):
    best_flux: float
    best_freq: float
    best_power: float


class JpaAutoOptimizeAdapter(
    BaseAdapter[JPAOptCfg, JpaAutoRunResult, JpaAutoAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = AutoOptimizeExp.persistence_tag()
    exp_cls = AutoOptimizeExp
    ExpCfg_cls: ClassVar[Any] = JPAOptCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        sweep_label = "Search bounds + relative resolution hints"
        sweep_tooltip = (
            "Start/stop set this axis's search bounds. Expts is a relative "
            "resolution hint compared with the other axes when allocating "
            "phase-1 flux slices; it is not a Cartesian or exact sample count."
        )
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(30.5)
            .device(
                "jpa_rf_dev",
                label="JPA RF device",
                default="",
                required=False,
            )
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
                    description="jpa flux device value search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .sweep(
                "jpa_freq",
                label="JPA pump freq (MHz)",
                default=custom(
                    jpa_freq_sweep_seed,
                    description="jpa pump frequency search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .sweep(
                "jpa_power",
                label="JPA pump power (dBm)",
                default=custom(
                    jpa_power_sweep_seed,
                    description="jpa pump power search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .int(
                "num_points",
                label="Global optimizer points",
                default=_JPA_AUTO_NUM_POINTS_DEFAULT,
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(1000)
            .rounds(1)
            .build()
        )

    def validate_run_request(
        self, req: RunRequest, raw_cfg: Mapping[str, ExperimentCfgValue]
    ) -> None:
        # Pure local cfg/schema validation; no device availability/settings checks.
        self.build_exp_cfg(raw_cfg, req)

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> JpaAutoRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, AutoOptimizeExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaAutoRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaAutoAnalyzeResult:
        analysis = AutoOptimizeExp().analyze(req.run_result, None, plots=plots)
        _relabel_auto_figure(plots["fit"])
        return JpaAutoAnalyzeResult(
            best_flux=analysis.best_flux,
            best_freq=analysis.best_freq,
            best_power=analysis.best_power,
        )

    def get_writeback_items(
        self, req: WritebackRequest[JpaAutoRunResult, JpaAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="best_jpa_flux",
                description="Best JPA flux device value",
                proposed_value=result.best_flux,
            ),
            ParameterWriteback(
                target_name="best_jpa_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="best_jpa_power",
                description="Best JPA pump power (dBm)",
                proposed_value=result.best_power,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_auto_{time.strftime('%m%d')}"
