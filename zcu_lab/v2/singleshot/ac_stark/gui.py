from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_chi_kappa,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.ac_stark.core import (
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)
from zcu_lab.v2.singleshot.ac_stark.form import make_form

# Domain analysis returns a numeric coefficient and publishes its named fit. The AC-Stark
# coefficient is written back to the parameter context (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]


@dataclass
class SsAcStarkAnalyzeResult(AnalyzeResultBase):
    ac_stark_coeff: float


class SsAcStarkAdapter(
    BaseAdapter[
        AcStarkCfg,
        SsAcStarkRunResult,
        SsAcStarkAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = AcStarkExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[AcStarkCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = AcStarkExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: AcStarkCfg,
        *,
        context: RunContext,
    ) -> SsAcStarkRunResult:
        del req
        result = AcStarkExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[SsAcStarkRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsAcStarkAnalyzeResult:
        # Required chi/kappa and optional readout correction/cutoff use this
        # Run's fixed capture. The core derives eta and owns fit defaults.
        chi, kappa = read_chi_kappa(req.parameters)
        confusion = read_confusion_matrix(req.parameters)
        cutoff = read_analysis_float(req.parameters, "$qubit.ext.cutoff")
        analysis = AcStarkExp().analyze(
            req.run_result,
            AcStarkAnalyzeOptions(
                chi=chi, kappa=kappa, confusion_matrix=confusion, cutoff=cutoff
            ),
            plots=plots,
        )
        return SsAcStarkAnalyzeResult(ac_stark_coeff=analysis.ac_stark_coeff)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsAcStarkRunResult, SsAcStarkAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``ac_stark_coeff`` per single_qubit.md:3329.
        return [
            ParameterWriteback(
                target_name="$qubit.ext.ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_stark_coeff,
            ),
        ]
