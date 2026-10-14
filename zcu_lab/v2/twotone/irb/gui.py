"""Measure GUI attachment for paired reference/interleaved randomized benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg, IRBResult

from .form import make_form

IRBRunResult: TypeAlias = RunRecord[IRBCfg, IRBResult]


@dataclass
class IRBAnalyzeResult(AnalyzeResultBase):
    """Unitless decay/error estimates, statistical confidence and pairing counts."""

    p_reference: float
    p_interleaved: float
    reference_epc: float
    gate_error: float
    gate_fidelity: float
    fidelity_ci_low: float
    fidelity_ci_high: float
    n_paired_seeds: int
    n_paired_rounds: int
    bootstrap_successes: int
    warning: str


class IRBAdapter(BaseAdapter[IRBCfg, IRBRunResult, IRBAnalyzeResult, NoAnalyzeParams]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[IRBCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    experiment_tag: ClassVar[str | None] = IRB_Exp.persistence_tag()

    exp_cls = IRB_Exp
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Paired single-qubit IRB: reference and interleaved arms share random "
            "Cliffords, each with its own full inverse. Depth counts random Cliffords; "
            "interleaved inserts one target per Clifford. Hardware sweeps depth and "
            "repetitions; software visits rounds, seeds and alternating arm order. Raw "
            "IQ has fixed axes round/seed/arm/depth, arm 0=reference, 1=interleaved."
        ),
        expects_md=(
            "Depth runs 0 to 100 random Cliffords over 21 points, seed 42 and 20 paired "
            "sequences. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the "
            "affected fields empty and Invalid; no numeric fallback."
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
            "No writeback. Fits both A*p**depth+B curves on a common IQ projection. "
            "Reports target error=(1-p_interleaved/p_reference)/2 and fidelity. 95% "
            "paired-seed bootstrap is statistical only; coherent or gate-dependent "
            "noise can bias this estimate. Nonphysical estimates are reported "
            "unclipped."
        ),
        recommended=(
            "Default target X90. Use at least 4 distinct depths spanning decay and 20 "
            "or more seeds for uncertainty. Start with a short execution check. "
            "Analysis uses complete round/seed pairs; unfinished raw IQ is still saved."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: IRBCfg,
        *,
        context: RunContext,
    ) -> IRBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        del req
        return RunRecord(cfg=cfg, result=IRB_Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[IRBRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> IRBAnalyzeResult:
        """Fit the saved run's mean IQ decay and publish its fit figure."""
        result = IRB_Exp().analyze(req.run_result, None, plots=plots)
        return IRBAnalyzeResult(
            p_reference=result.p_reference,
            p_interleaved=result.p_interleaved,
            reference_epc=result.reference_epc,
            gate_error=result.gate_error,
            gate_fidelity=result.gate_fidelity,
            fidelity_ci_low=result.fidelity_ci_low,
            fidelity_ci_high=result.fidelity_ci_high,
            n_paired_seeds=result.n_paired_seeds,
            n_paired_rounds=result.n_paired_rounds,
            bootstrap_successes=result.bootstrap_successes,
            warning=result.warning,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        """Return the qubit and date stem used by canonical result persistence."""
        return f"{ctx.qub_name}_irb_{time.strftime('%m%d')}"
