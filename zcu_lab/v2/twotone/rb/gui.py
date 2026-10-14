"""Measure GUI attachment for single-qubit Clifford randomized benchmarking."""

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
from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBCfg

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


@dataclass
class RBAnalyzeResult(AnalyzeResultBase):
    """Single-qubit error per Clifford and mean Clifford fidelity (unitless)."""

    epc: float
    fidelity: float


class RBAdapter(BaseAdapter[RBCfg, RBRunResult, RBAnalyzeResult, NoAnalyzeParams]):
    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    experiment_tag: ClassVar[str | None] = RB_Exp.persistence_tag()

    exp_cls = RB_Exp
    ExpCfg_cls: ClassVar[type[RBCfg]] = RBCfg
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Declare calibrated modules, integer depths, random seeds and averaging."""
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
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
