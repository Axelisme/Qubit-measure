from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.ac_stark.core import (
    AcStarkAnalysis,
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)
from zcu_lab.v2.singleshot.ac_stark.form import make_form
from zcu_lab.v2.support.measure.singleshot_shared import (
    read_analysis_float,
    read_chi_kappa,
    read_confusion_matrix,
)

# Domain analysis returns a numeric coefficient and publishes its named fit. The AC-Stark
# coefficient is written back to the parameter context (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]


def _analyze(
    req: AnalyzeRequest[SsAcStarkRunResult, None], *, plots: Plots
) -> AcStarkAnalysis:
    chi, kappa = read_chi_kappa(req.parameters)
    confusion = read_confusion_matrix(req.parameters)
    cutoff = read_analysis_float(req.parameters, "$qubit.ext.cutoff")
    return AcStarkExp().analyze(
        req.run_result,
        AcStarkAnalyzeOptions(
            chi=chi, kappa=kappa, confusion_matrix=confusion, cutoff=cutoff
        ),
        plots=plots,
    )


def _writeback(
    req: WritebackRequest[SsAcStarkRunResult, AcStarkAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$qubit.ext.ac_stark_coeff",
            description="AC-Stark coefficient (photon number per gain²)",
            proposed_value=req.analyze_result.ac_stark_coeff,
        ),
    ]


def _run(
    cfg: AcStarkCfg, *, context: RunContext
) -> RunRecord[AcStarkCfg, AcStarkResult]:
    return RunRecord(cfg=cfg, result=AcStarkExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AcStarkExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_writeback),
)
