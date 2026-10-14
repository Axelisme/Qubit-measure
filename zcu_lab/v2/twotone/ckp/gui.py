"""CKP GUI adapter for ground/excited-resolved two-tone spectroscopy."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

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

from zcu_lab.v2.twotone.ckp.core import CKP_Cfg, CKP_Exp, CKP_Result

from .form import make_form

CKPRunResult: TypeAlias = RunRecord[CKP_Cfg, CKP_Result]


@dataclass
class CKPAnalyzeResult(AnalyzeResultBase):
    chi: float
    kappa: float
    res_freq: float


class CKPAdapter(BaseAdapter[CKP_Cfg, CKPRunResult, CKPAnalyzeResult, NoAnalyzeParams]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[CKP_Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = CKP_Exp.persistence_tag()
    exp_cls = CKP_Exp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "CKP ground/excited-resolved spectroscopy: prepares the qubit in g and e, "
            "drives resonator and qubit probe tones concurrently over a two-dimensional "
            "frequency sweep, and fits the two resonator branches to extract dispersive "
            "shift chi, linewidth kappa, and the readout frequency. Runs on real "
            "hardware."
        ),
        expects_md=(
            "Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, "
            "resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the "
            "fixed capture. Missing required calibration remains Invalid; no numeric "
            "fallback. Resonator sweep is resonator.freq ± 1.5 * resonator.kappa. Qubit "
            "sweep runs qubit.freq - 10 to qubit.freq + 5 MHz. Probe timing scales "
            "inversely with resonator.kappa."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a fresh readout template when "
            "unset or dangling. Optional qubit.module.reset stays disabled without a "
            "source. Library references remain selectable; sweeps overwrite their "
            "controlled fields only at Run. Pi pulse uses qubit.module.x180 with a "
            "pi_pulse template fallback; both probe pulses are fresh inline templates."
        ),
        typical_writeback=(
            "Proposes fitted chi into '$qubit.ext.chi', fitted kappa into "
            "'$resonator.kappa', and fitted readout frequency into '$resonator.freq'. "
            "No ModuleLibrary writeback."
        ),
        recommended=(
            "Run only after qubit frequency, pi pulse, resonator frequency, and "
            "resonator linewidth are calibrated. Review both g/e maps and fitted "
            "branches before applying writeback; a branch-selection or fit failure can "
            "produce plausible-looking scalar values."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: CKP_Cfg,
        *,
        context: RunContext,
    ) -> CKPRunResult:
        del req
        result = CKP_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CKPRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> CKPAnalyzeResult:
        analysis = CKP_Exp().analyze(req.run_result, None, plots=plots)
        return CKPAnalyzeResult(
            chi=analysis.chi,
            kappa=analysis.kappa,
            res_freq=analysis.res_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[CKPRunResult, CKPAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="$qubit.ext.chi",
                description="Dispersive shift chi (MHz)",
                proposed_value=result.chi,
            ),
            ParameterWriteback(
                target_name="$resonator.kappa",
                description="Resonator linewidth kappa (MHz)",
                proposed_value=result.kappa,
            ),
            ParameterWriteback(
                target_name="$resonator.freq",
                description="Readout frequency (MHz)",
                proposed_value=result.res_freq,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ckp_{time.strftime('%m%d')}"
