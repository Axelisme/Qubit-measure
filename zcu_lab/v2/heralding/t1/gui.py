"""GUI wiring for single-qubit heralded T1."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions

from .core import (
    HeraldingT1Analysis,
    HeraldingT1Cfg,
    HeraldingT1Exp,
    HeraldingT1Result,
)
from .form import make_form


def run(
    cfg: HeraldingT1Cfg, *, context: RunContext
) -> RunRecord[HeraldingT1Cfg, HeraldingT1Result]:
    """Retain the classifier and acquisition configuration with the result."""
    return RunRecord(cfg=cfg, result=HeraldingT1Exp().run(cfg, context=context))


def analyze(
    req: AnalyzeRequest[RunRecord[HeraldingT1Cfg, HeraldingT1Result], T1AnalyzeOptions],
    *,
    plots: Plots,
) -> HeraldingT1Analysis:
    """Fit the post-selected data with nonfinite points removed."""
    return HeraldingT1Exp().analyze(req.run_result, req.analyze_params, plots=plots)


def writeback(
    req: WritebackRequest[
        RunRecord[HeraldingT1Cfg, HeraldingT1Result], HeraldingT1Analysis
    ],
) -> list[ParameterWriteback]:
    """Propose T1 in us for the captured qubit; applying remains explicit."""
    return [
        ParameterWriteback(
            target_name="$qubit.t1",
            description="Heralded T1 relaxation time (us)",
            proposed_value=req.analyze_result.t1,
            stderr=req.analyze_result.t1_err,
        )
    ]


DEFINITION = Adapter(
    spec=HeraldingT1Exp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: T1AnalyzeOptions(),
        writeback=writeback,
    ),
)
