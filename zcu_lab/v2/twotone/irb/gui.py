"""Measure GUI attachment for paired reference/interleaved randomized benchmarking."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import SweepDefault, scaled_path_seed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
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

    experiment_tag: ClassVar[str | None] = IRB_Exp.persistence_tag()

    exp_cls = IRB_Exp
    ExpCfg_cls: ClassVar[type[IRBCfg]] = IRBCfg
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare calibrated modules, integer depths, random seeds and averaging."""
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
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
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=200.0)
            )
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
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
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
