from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SampleWriteback,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalyzeOptions,
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)

from .form import make_form

logger = logging.getLogger(__name__)

T2EchoRunResult: TypeAlias = RunRecord[T2EchoCfg, T2EchoResult]


@dataclass
class T2EchoAnalyzeParams:
    fit_method: Annotated[Literal["fringe", "decay"], ParamMeta(label="Fit method")] = (
        "fringe"
    )
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


@dataclass
class T2EchoAnalyzeResult(AnalyzeResultBase):
    t2e: float
    t2e_err: float
    fit_quality: dict[str, dict[str, object]] | None = None


class T2EchoAdapter(
    BaseAdapter[
        T2EchoCfg,
        T2EchoRunResult,
        T2EchoAnalyzeResult,
        T2EchoAnalyzeParams,
    ]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T2EchoCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = T2EchoExp.persistence_tag()
    exp_cls = T2EchoExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T2 echo (spin/Hahn echo): a pi/2 – delay – pi – delay – pi/2 sequence with "
            "the total free-evolution delay swept; the refocusing pi pulse cancels "
            "low-frequency dephasing, and the decay fit yields the echo coherence time "
            "T2E. Runs on real hardware. Run after both pi and pi/2 pulses are "
            "calibrated; T2E is typically longer than the Ramsey T2*."
        ),
        expects_md=(
            "Total delay runs 0 to 1.5 * qubit.t2e us over 101 points. Detune ratio is "
            "fringes per sweep step; native MHz detune is ratio / step. Relax delay is "
            "5 * qubit.t1 us. Missing calibration leaves the affected fields empty and "
            "Invalid; no numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). Drive slots "
            "use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. "
            "Unset or dangling slots use the named template, not a legacy blank. Key "
            "choices remain available. Inline templates read qubit.freq, "
            "qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and "
            "resonator.ext.time_of_fly + 0.05 us; missing required calibration remains "
            "Invalid."
        ),
        typical_writeback=(
            "Proposes fitted qubit.t2e (us) to the captured qubit and working point, "
            "including analysis of a source without cfg. No frequency or module "
            "writeback; later focus changes do not retarget it. An independent "
            "t2e_us/t2e_err_us sample uses the captured Run bias when device evidence "
            "is available."
        ),
        recommended=(
            "Enable Fit phase offset to fit a free fringe phase; it defaults to fixed "
            "phase and has no effect for decay-only analysis. Set the 'Detune ratio "
            "(fringes/step)' cfg knob (default 0.1) — it is the number of fringe "
            "periods per delay-sweep step; the absolute detune (MHz) applied to the "
            "final pi/2 pulse phase is detune_ratio / sweep step, leaving a residual "
            "oscillation on the echo decay. Analysis 'Fit method' defaults to 'fringe' "
            "(matching the nonzero default ratio) so the fringe frequency is fit; the "
            "detune is computed but NOT written back. Set the ratio to 0 and switch "
            "'Fit method' to 'decay' for a standard echo (pure exponential decay → "
            "T2E). A total-delay sweep reaching a few T2E lets the echo decay; the "
            "first sweep point is dropped in the fit."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: T2EchoCfg,
        *,
        context: RunContext,
    ) -> T2EchoRunResult:
        del req
        result = self.exp_cls().run(cfg, context=context)
        logger.info("T2Echo true detune: %s MHz", result.true_activate_detune)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[T2EchoRunResult, T2EchoAnalyzeParams],
        *,
        plots: Plots,
    ) -> T2EchoAnalyzeResult:
        params = req.analyze_params
        analysis = T2EchoExp().analyze(
            req.run_result,
            T2EchoAnalyzeOptions(
                fit_method=params.fit_method, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        return T2EchoAnalyzeResult(
            t2e=analysis.t2e,
            t2e_err=analysis.t2e_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2EchoRunResult, T2EchoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose fitted T2E in us and a sample for the Run's resolved $qubit.

        req supplies the result and its captured context; source.cfg may be None.
        WritebackService resolves the role and destination on req.ctx.parameters
        and rejects a missing capture or unbound role. Without a capture,
        retain parameter proposals only so loaded data can still be analyzed.
        With a capture, unbound $qubit raises RoleResolutionError. The service
        adds bias coordinates or omits the sample with a public reason. This
        Run leaves devices fixed; no live lookup, device access, frequency/module
        update or write occurs here.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t2e",
                description="T2 Echo time (us)",
                proposed_value=result.t2e,
            ),
        ]
        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="T2 Echo sample at the captured bias",
                    proposed_value={
                        "t2e_us": result.t2e,
                        "t2e_err_us": result.t2e_err,
                    },
                )
            )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t2echo_{time.strftime('%m%d')}"
