"""Recording analysis/post adapters with distinct named figures and proposals."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.types import (
    PostAnalyzeResultBase,
    T_PostAnalyzeResult,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._synthetic import EmptyParams


@dataclass
class PaneParams:
    initial_state: str = "ground"
    length_ratio: float = 1.0


@dataclass
class PaneResult(AnalyzeResultBase):
    initial_state: str


@dataclass
class PostResult(PostAnalyzeResultBase):
    confusion: NDArray[np.float64]


SavingRun = RunRecord[SavingCfg, SavingResult]


def PaneAdapter_make_form(ctx: FormContext) -> Form[SavingCfg]:
    del ctx
    return Form(SavingCfg, {})


def PaneAdapter_analyze(
    req: AnalyzeRequest[SavingRun, PaneParams], *, plots: Plots
) -> PaneResult:
    _, ax = plots.subplots("fit")
    ax.plot([0, 1], [0, 1])
    return PaneResult(req.analyze_params.initial_state)


def PaneAdapter_get_post_analyze_params(
    source, analyze_result: PaneResult, ctx: SessionEnv
) -> EmptyParams:
    return EmptyParams()


def PaneAdapter_post_analyze(
    req: PostAnalyzeRequest[SavingRun, PaneResult, EmptyParams], *, plots: Plots
) -> PostResult:
    if req.analyze_result.initial_state != "excited":
        raise ValueError("Post must receive the adopted primary result")
    _, ax = plots.subplots("post")
    ax.plot([0, 1], [1, 0])
    return PostResult(np.eye(3))


def PaneAdapter_get_writeback_items(
    req: WritebackRequest[SavingRun, PaneResult],
) -> Sequence[ParameterWriteback]:
    return [
        ParameterWriteback(f"$qubit.ext.readout_cal.{name}", name, value)
        for name, value in (
            ("fidelity", 1.0),
            ("sigma", 0.2),
            ("g_center", [0.0, 0.0]),
            ("e_center", [1.0, 1.0]),
        )
    ]


def PaneAdapter_get_post_writeback_items(
    req: PostWritebackRequest[SavingRun, PaneResult, T_PostAnalyzeResult],
) -> Sequence[ParameterWriteback]:
    """Build pane proposals from PostResult; reject other payloads with TypeError."""
    result = req.post_analyze_result
    if not isinstance(result, PostResult):
        raise TypeError("PaneAdapter post writeback requires PostResult")
    return [
        ParameterWriteback("$qubit.ext.readout_cal.radius", "Radius", 0.5),
        ParameterWriteback(
            "$qubit.ext.readout_cal.confusion_matrix",
            "Matrix",
            result.confusion.tolist(),
        ),
    ]


PaneAdapter = Adapter(
    spec=replace(SavingCore.AXES_SPEC, tag="test/saving"),
    make_form=PaneAdapter_make_form,
    run=SavingCore().run,
    guide=None,
    analysis=Analysis(
        fn=PaneAdapter_analyze,
        options=lambda source, env: PaneParams(),
        writeback=PaneAdapter_get_writeback_items,
    ),
    post_analysis=PostAnalysis(
        fn=PaneAdapter_post_analyze,
        options=PaneAdapter_get_post_analyze_params,
        writeback=PaneAdapter_get_post_writeback_items,
    ),
)
