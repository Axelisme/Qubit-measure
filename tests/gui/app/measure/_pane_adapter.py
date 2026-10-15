"""Recording analysis/post adapters with distinct named figures and proposals."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    PostAnalyzeRequest,
    PostWritebackRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
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


class PaneAdapter(BaseAdapter[SavingCfg, SavingRun, PaneResult, PaneParams]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        post_analysis=True, requires_soc=False, load_data=True
    )
    exp_cls = SavingCore

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SavingCfg]:
        del ctx
        return Form(SavingCfg, {})

    experiment_tag: ClassVar[str | None] = "test/saving"

    @classmethod
    def analyze_params_cls(cls) -> type[PaneParams]:
        return PaneParams

    def analyze(
        self, req: AnalyzeRequest[SavingRun, PaneParams], *, plots: Plots
    ) -> PaneResult:
        _, ax = plots.subplots("fit")
        ax.plot([0, 1], [0, 1])
        return PaneResult(req.analyze_params.initial_state)

    def get_post_analyze_params(
        self, analyze_result: PaneResult, ctx: SessionEnv
    ) -> EmptyParams:
        return EmptyParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[SavingRun, PaneResult, EmptyParams],
        *,
        plots: Plots,
    ) -> PostResult:
        if req.analyze_result.initial_state != "excited":
            raise ValueError("Post must receive the adopted primary result")
        _, ax = plots.subplots("post")
        ax.plot([0, 1], [1, 0])
        return PostResult(np.eye(3))

    def get_writeback_items(
        self, req: WritebackRequest[SavingRun, PaneResult]
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

    def get_post_writeback_items(
        self, req: PostWritebackRequest[SavingRun, PaneResult, T_PostAnalyzeResult]
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
