"""Measure GUI attachment for single-qubit Clifford randomized benchmarking."""

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

from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBCfg

from .form import make_form

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


@dataclass
class RBAnalyzeResult(AnalyzeResultBase):
    """Single-qubit error per Clifford and mean Clifford fidelity (unitless)."""

    epc: float
    fidelity: float


class RBAdapter(BaseAdapter[RBCfg, RBRunResult, RBAnalyzeResult, NoAnalyzeParams]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[RBCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    experiment_tag: ClassVar[str | None] = RB_Exp.persistence_tag()

    exp_cls = RB_Exp
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-qubit Clifford RB: each independent random seed supplies a shared "
            "prefix across depths, followed by its inverse and readout. Depth counts "
            "Cliffords, not physical pulses. Z rotations are virtual. Runs one hardware "
            "program per seed, sweeping depth within it."
        ),
        expects_md=(
            "Depth runs 0 to 100 Cliffords over 21 points, seed 42 and 20 sequences. "
            "Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected "
            "fields empty and Invalid; no numeric fallback."
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
            "No writeback. Fits mean projected IQ to A*p**depth+B; reports epc=(1-p)/2 "
            "and fidelity=1-epc per Clifford, not per physical gate."
        ),
        recommended=(
            "Calibrate frequency and pulse amplitudes first. Start with few seeds and "
            "shallow depths to verify execution, then use independent seeds and depths "
            "spanning the decay. Seed controls reproducibility. Review residuals and "
            "seed variation before interpreting fidelity."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: RBCfg,
        *,
        context: RunContext,
    ) -> RBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        del req
        return RunRecord(cfg=cfg, result=RB_Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[RBRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> RBAnalyzeResult:
        """Fit the saved run's mean IQ decay and publish its fit figure."""
        result = RB_Exp().analyze(req.run_result, None, plots=plots)
        return RBAnalyzeResult(epc=result.epc, fidelity=result.fidelity)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        """Return the qubit and date stem used by canonical result persistence."""
        return f"{ctx.qub_name}_rb_{time.strftime('%m%d')}"
