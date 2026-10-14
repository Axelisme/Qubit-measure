"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure
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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
)
from zcu_lab.v2.jpa.auto_optimize.form import make_form

JpaAutoRunResult = RunRecord[JPAOptCfg, JPAOptimizeResult]


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

    @staticmethod
    def make_form(ctx: FormContext) -> Form[JPAOptCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA joint optimization: with the qubit prepared in g and e (a pi "
            "pulse toggles it), a multi-phase optimizer searches the JPA flux "
            "device value, pump frequency and pump power jointly to maximize "
            "the g/e signal difference. Runs on real hardware and commands "
            "the selected RF and flux devices at every evaluated point, "
            "leaving them at the last evaluated point. WARNING: review the "
            "selected devices and the search bounds before running — the "
            "seeded bounds are bring-up defaults, not certified safety "
            "limits."
        ),
        expects_md=(
            "Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. "
            "Search bounds use $jpa.flux, $jpa.pump_freq and $jpa.pump_power. The "
            "required JPA flux device resolves flux_dev once at Open or Reset; select "
            "the connected RF device in the form. RF defaults capture jpa_rf_dev "
            "at Open or Reset; an unbound alias leaves the optional field empty. "
            "flux_unit starts as A with no "
            "conversion. Missing required aliases or calibration leave the form "
            "Invalid."
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
            "Proposes the signal-maximizing flux device value, pump frequency "
            "and pump power into three separate drafts: $jpa.flux in "
            "$jpa.flux_unit coordinates, $jpa.pump_freq (MHz) and "
            "$jpa.pump_power (dBm). You can accept, cancel or retarget each "
            "independently. Nothing is written without your acceptance. "
            "Writeback never commands a device or changes the current-source value."
        ),
        recommended=(
            "For each sweep, start/stop are SEARCH BOUNDS and expts is a "
            "RELATIVE RESOLUTION HINT — expts is not a Cartesian or exact "
            "sample count. The optimizer compares all three hints when "
            "allocating its phase-1 flux slices and points per slice. "
            "'Global optimizer points' (num_points, at least 4, default "
            "1001) remains the total evaluation budget: raise it for wider "
            "bounds, lower it for a quick search. Analysis reports the best "
            "evaluated point with no tuning knobs."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: JPAOptCfg,
        *,
        context: RunContext,
    ) -> JpaAutoRunResult:
        del req
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
                target_name="$jpa.flux",
                description="Best JPA flux device value",
                proposed_value=result.best_flux,
            ),
            ParameterWriteback(
                target_name="$jpa.pump_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="$jpa.pump_power",
                description="Best JPA pump power (dBm)",
                proposed_value=result.best_power,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_auto_{time.strftime('%m%d')}"
