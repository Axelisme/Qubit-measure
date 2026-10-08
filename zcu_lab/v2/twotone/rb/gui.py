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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBCfg

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


@dataclass
class RBAnalyzeResult(AnalyzeResultBase):
    """Single-qubit error per Clifford and mean Clifford fidelity (unitless)."""

    epc: float
    fidelity: float


class RBAdapter(BaseAdapter[RBCfg, RBRunResult, RBAnalyzeResult, NoAnalyzeParams]):
    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    exp_cls = RB_Exp
    ExpCfg_cls: ClassVar[type[RBCfg]] = RBCfg
    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-qubit Clifford RB: each independent random seed supplies a "
            "shared prefix across depths, followed by its inverse and readout. "
            "Depth counts Cliffords, not physical pulses. Z rotations are virtual. "
            "Runs one hardware program per seed, sweeping depth within it."
        ),
        expects_md="Uses t1 (us) for a 5*t1 recovery delay, fallback 200 us.",
        expects_ml=(
            "Requires calibrated X90 (pi2_amp/pi2_len), X180 (pi_amp/pi_len) "
            "and readout modules. Verify both drive frequencies and channels. "
            "Identity uses X90 at zero gain. Reset is optional."
        ),
        typical_writeback=(
            "No writeback. Fits mean projected IQ to A*p**depth+B; reports "
            "epc=(1-p)/2 and fidelity=1-epc per Clifford, not per physical gate."
        ),
        recommended=(
            "Calibrate frequency and pulse amplitudes first. Start with few "
            "seeds and shallow depths to verify execution, then use independent "
            "seeds and depths spanning the decay. Seed controls reproducibility. "
            "Review residuals and seed variation before interpreting fidelity."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare calibrated modules, integer depths, random seeds and averaging."""
        return (
            MeasureCfgBuilder()
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
    ) -> RBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        cfg = self.build_exp_cfg(raw_cfg, req)
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
