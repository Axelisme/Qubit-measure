"""Mux GE primary/post hooks using paired raw shots and captured destinations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    T_PostAnalyzeResult,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.mux_results import captured_qubits, indexed_summary
from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions, GEPostAnalyzeOptions
from zcu_lab.v2.singleshot.ge.gui import GEAnalyzeParams, GEPostAnalyzeParams

from .analysis import (
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
)
from .core import MuxGECfg, MuxGEExp, MuxGEResult

MuxGERunResult: TypeAlias = RunRecord[MuxGECfg, MuxGEResult]


@dataclass(frozen=True)
class MuxGEAnalyzeResult(AnalyzeResultBase):
    """Primary per-index outcomes and full source names, absent without cfg."""

    outcomes: tuple[MuxGEAnalysis | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present same-source GE calibration/quality or each retained failure."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                primary = outcome.primary
                values = {
                    "initial_state": primary.initial_state,
                    "fidelity": primary.fidelity,
                    "theta": primary.theta,
                    "threshold": primary.threshold,
                    "ge_s": primary.ge_s,
                    "init_pops": primary.init_pops.tolist(),
                    "fit_quality": fit_quality_summary(
                        primary.fit_quality,
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


@dataclass(frozen=True)
class MuxGEPostAnalyzeResult(PostAnalyzeResultBase):
    """Per-index single-qubit post diagnostics and names from the adopted primary."""

    outcomes: tuple[MuxGEPostAnalysis | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present radius/confusion diagnostics, not a joint assignment matrix."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                values = {
                    "ge_radius": outcome.post.confusion.radius,
                    "confusion": outcome.post.confusion.matrix.tolist(),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


class MuxGEAdapter(
    BaseAdapter[MuxGECfg, MuxGERunResult, MuxGEAnalyzeResult, GEAnalyzeParams]
):
    """Acquire paired mux shots, analyze via GE owner and propose without Apply."""

    experiment_tag: ClassVar[str | None] = MuxGEExp.persistence_tag()
    exp_cls = MuxGEExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    def run(
        self, req: RunRequest, cfg: MuxGECfg, *, context: RunContext
    ) -> MuxGERunResult:
        """Pair the original validated controls with this operation's raw result."""
        del req
        return RunRecord(cfg, MuxGEExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxGERunResult, GEAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxGEAnalyzeResult:
        """Use core GE options on the explicit source, publishing fit_i figures."""
        params = req.analyze_params
        outcomes = MuxGEExp().analyze(
            req.run_result,
            MuxGEAnalyzeOptions(
                default=GEAnalyzeOptions(
                    initial_state=params.initial_state,
                    backend=params.backend,
                    logscale=params.logscale,
                    align_t1=params.align_t1,
                    length_ratio=params.length_ratio,
                )
            ),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
        return MuxGEAnalyzeResult(outcomes, names)

    def get_post_analyze_params(
        self, analyze_result: MuxGEAnalyzeResult, ctx: SessionEnv
    ) -> GEPostAnalyzeParams:
        """Return the existing parameter-free GE post policy for the adopted fit."""
        del analyze_result, ctx
        return GEPostAnalyzeParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[
            MuxGERunResult, MuxGEAnalyzeResult, GEPostAnalyzeParams
        ],
        *,
        plots: Plots,
    ) -> MuxGEPostAnalyzeResult:
        """Apply the same-source adopted primary via GE owner, publishing post_i."""
        outcomes = MuxGEExp().post_analyze(
            req.run_result,
            req.analyze_result.outcomes,
            MuxGEPostOptions(default=GEPostAnalyzeOptions()),
            plots=plots,
        )
        return MuxGEPostAnalyzeResult(outcomes, req.analyze_result.names)

    def get_writeback_items(
        self, req: WritebackRequest[MuxGERunResult, MuxGEAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose complete successful tones' primary IQ calibrations, never Apply.

        Resolve destinations only from this Run/Load capture; no capture means
        no proposals. Validate the binding against any source cfg. Preserve GE's
        calibration validity gate. Failed/partial rows contribute no items.
        """
        snapshot = req.run_result.cfg
        names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
        qubits = captured_qubits(req.ctx.parameters, names)
        if qubits is None:
            return []
        items: list[WritebackItem] = []
        for row, outcome in enumerate(req.analyze_result.outcomes):
            if (
                isinstance(outcome, MuxFitFailure)
                or not np.isfinite(req.run_result.result.signals[row]).all()
            ):
                continue
            primary = outcome.primary
            primary.validate_calibration()
            qubit = qubits[outcome.index]
            for field, value, description in (
                ("fidelity", primary.fidelity, "Single-shot assignment fidelity"),
                ("sigma", primary.ge_s, "Single-shot IQ cluster width (s)"),
                ("g_center", primary.g_center, "Single-shot ground IQ center"),
                ("e_center", primary.e_center, "Single-shot excited IQ center"),
            ):
                items.append(
                    ParameterWriteback(
                        target_name=f"{qubit}.ext.readout_cal.{field}",
                        description=description,
                        proposed_value=value,
                    )
                )
        return items

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[
            MuxGERunResult, MuxGEAnalyzeResult, T_PostAnalyzeResult
        ],
    ) -> Sequence[WritebackItem]:
        """Propose only same-source successful GE radius/confusion diagnostics.

        Missing capture returns no items. Invalid binding or wrong post result
        raises; failed/partial rows contribute nothing. Targets retain GE's 3x3
        diagnostic policy. No current Form/library lookup or Apply occurs here.
        """
        result = req.post_analyze_result
        if not isinstance(result, MuxGEPostAnalyzeResult):
            raise TypeError("Mux GE post proposals require MuxGEPostAnalyzeResult")
        snapshot = req.run_result.cfg
        names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
        qubits = captured_qubits(req.ctx.parameters, names)
        if qubits is None:
            return []
        items: list[WritebackItem] = []
        for row, outcome in enumerate(result.outcomes):
            if (
                isinstance(outcome, MuxFitFailure)
                or not np.isfinite(req.run_result.result.signals[row]).all()
            ):
                continue
            qubit = qubits[outcome.index]
            items.extend(
                [
                    ParameterWriteback(
                        target_name=f"{qubit}.ext.readout_cal.radius",
                        description="Single-shot classification radius",
                        proposed_value=outcome.post.confusion.radius,
                    ),
                    ParameterWriteback(
                        target_name=f"{qubit}.ext.readout_cal.confusion_matrix",
                        description="Single-shot 3x3 confusion matrix",
                        proposed_value=outcome.post.confusion.matrix.tolist(),
                    ),
                ]
            )
        return items
