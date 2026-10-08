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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg, IRBResult

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
    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    exp_cls = IRB_Exp
    ExpCfg_cls: ClassVar[type[IRBCfg]] = IRBCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Paired single-qubit IRB: reference and interleaved arms share random "
            "Cliffords, each with its own full inverse. Depth counts random Cliffords; "
            "interleaved inserts one target per Clifford. Hardware sweeps depth and "
            "repetitions; software visits rounds, seeds and alternating arm order. "
            "Raw IQ has fixed axes round/seed/arm/depth, arm 0=reference, 1=interleaved."
        ),
        expects_md="Uses 5*t1 relaxation (us), fallback 200 us.",
        expects_ml="Uses calibrated X90/X180 and readout modules. Y gates use phase-shifted X pulses.",
        typical_writeback=(
            "No writeback. Fits both A*p**depth+B curves on a common IQ projection. "
            "Reports target error=(1-p_interleaved/p_reference)/2 and fidelity. "
            "95% paired-seed bootstrap is statistical only; coherent or gate-dependent "
            "noise can bias this estimate. Nonphysical estimates are reported unclipped."
        ),
        recommended=(
            "Default target X90. Use at least 4 distinct depths spanning decay and "
            "20 or more seeds for uncertainty. Start with a short execution check. "
            "Analysis uses complete round/seed pairs; unfinished raw IQ is still saved."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare calibrated modules, integer depths, random seeds and averaging."""
        return (
            MeasureCfgBuilder()
            .choice(
                "target_gate",
                label="Target gate",
                choices=("X90", "X180", "Y90", "Y180"),
                default="X90",
            )
            .reset(optional=True)
            .pulse("X90_pulse", role_id="pi2_pulse", label="X90 Pulse")
            .pulse("X180_pulse", role_id="pi_pulse", label="X180 Pulse")
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=200.0))
            .sweep(
                "depth",
                label="Clifford depth",
                decimals=0,
                default=SweepDefault(start=0, stop=100, expts=21),
            )
            .int("seed", label="Random seed", default=42)
            .int("n_seeds", label="Number of sequences", default=20)
            .reps(1000)
            .rounds(1)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> IRBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        cfg = self.build_exp_cfg(raw_cfg, req)
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
