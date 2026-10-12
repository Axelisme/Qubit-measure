"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
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
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import alias_name_seed, custom
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.jpa.flux.core import FluxCfg, FluxExp, FluxResult

JpaFluxRunResult: TypeAlias = RunRecord[FluxCfg, FluxResult]

_JPA_FLUX_SWEEP_EXPTS = 101
# Bring-up survey: ±5e-3 around the centre, taken from the notebook's JPA flux
# sweep (single_qubit.md, -5e-3..5e-3) and compressed to 101 points. These are
# inspectable starting bounds, NOT safety certification — the operator must
# review device and sweep.
_JPA_FLUX_SEED_SPAN = 5.0e-3


def jpa_flux_sweep_seed(
    ctx: SessionEnv, *, expts: int = _JPA_FLUX_SWEEP_EXPTS
) -> SweepValue:
    """Store ±5e-3 around $jpa.flux without reading ctx calibration.

    Values use the selected device's working unit, not a unit conversion.
    Missing aliases/calibration become Invalid during cfg resolution.
    """
    del ctx  # Defaults retain expressions; the resource owns source reads.
    return SweepValue(
        start=EvalValue(expr=f"$jpa.flux - {_JPA_FLUX_SEED_SPAN}"),
        stop=EvalValue(expr=f"$jpa.flux + {_JPA_FLUX_SEED_SPAN}"),
        expts=expts,
    )


@dataclass
class JpaFluxAnalyzeResult(AnalyzeResultBase):
    best_flux: float


def _relabel_flux_figure(fig: Figure, best_flux: float) -> None:
    """Neutral JPA flux device value wording on the GUI review figure.

    The core analysis figure still labels its flux axis and the optimum legend
    with 'a.u.'; the GUI review figure must use the neutral 'JPA flux device
    value' vocabulary (A6) without claiming a physical-unit migration. The
    relabel happens at the adapter analysis boundary, in place, so the review
    figure wording is adapter-owned while the core figure stays untouched.
    """
    for ax in fig.axes:
        ax.set_xlabel("JPA flux device value")
        for line in ax.get_lines():
            label = line.get_label()
            if isinstance(label, str) and label.startswith("best JPA flux"):
                line.set_label(f"best JPA flux device value = {best_flux:.2g}")
        legend = ax.get_legend()
        if legend is not None:
            ax.legend()


class JpaFluxAdapter(
    BaseAdapter[FluxCfg, JpaFluxRunResult, JpaFluxAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FluxExp.persistence_tag()
    exp_cls = FluxExp
    ExpCfg_cls: ClassVar[Any] = FluxCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA flux-device calibration: with the qubit prepared in g and e "
            "(a pi pulse toggles it), sweeps the JPA flux device value and "
            "measures the g/e signal difference, so you can pick the flux "
            "device value that best enhances readout. Runs on real hardware. "
            "WARNING: review the selected JPA flux device and the sweep before "
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
            "Proposes the signal-maximizing JPA flux device value into "
            "parameter context 'best_jpa_flux' as a draft — it never writes it back "
            "without your acceptance, never commands the device from the "
            "writeback, and never touches 'cur_jpa_A'."
        ),
        recommended=(
            "Review the selected flux device and the sweep bounds before every "
            "run; the seeded range around the current centre is only a "
            "starting point. Analysis picks the peak of the absolute signal "
            "difference."
        ),
    )

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
    ) -> JpaFluxRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FluxExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaFluxRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaFluxAnalyzeResult:
        answer = FluxExp().analyze(req.run_result, None, plots=plots)
        _relabel_flux_figure(plots["fit"], answer.best_flux)
        return JpaFluxAnalyzeResult(best_flux=answer.best_flux)

    def get_writeback_items(
        self, req: WritebackRequest[JpaFluxRunResult, JpaFluxAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="best_jpa_flux",
                description="Best JPA flux device value",
                proposed_value=req.analyze_result.best_flux,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_flux_{time.strftime('%m%d')}"
