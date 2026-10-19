"""Measure adapter for constructive phase."""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import module_writeback

from .._rabi_scan import RabiScanResult
from .analysis import ConstructivePhaseAnalysis
from .core import ConstructivePhaseCfg, ConstructivePhaseExp
from .form import make_form

ConstructivePhaseRun: TypeAlias = RunRecord[ConstructivePhaseCfg, RabiScanResult]


class ConstructivePhaseAdapter(
    BaseAdapter[
        ConstructivePhaseCfg,
        ConstructivePhaseRun,
        ConstructivePhaseAnalysis,
        NoAnalyzeParams,
    ]
):
    """Fixed-capture acquisition, saved-source analysis and native restore."""

    exp_cls = ConstructivePhaseExp
    experiment_tag: ClassVar[str | None] = ConstructivePhaseExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ConstructivePhaseCfg]:
        """Capture editable pulses, GE centers and locked role/tone identities."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: ConstructivePhaseCfg, *, context: RunContext
    ) -> ConstructivePhaseRun:
        """Pair original cfg with full raw; caller checks cancel_signal errors."""
        del req
        return RunRecord(
            cfg=cfg, result=ConstructivePhaseExp().run(cfg, context=context)
        )

    def analyze(
        self,
        req: AnalyzeRequest[ConstructivePhaseRun, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Analyze only saved source cfg/raw, never the current Form or library."""
        return ConstructivePhaseExp().analyze(req.run_result, None, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[ConstructivePhaseRun, ConstructivePhaseAnalysis]
    ) -> Sequence[WritebackItem]:
        """Propose only F2 phase and its independent coupler slot from saved cfg.

        F2.phase is source F1.phase plus the identifiable maximum-rate relative
        phase. Preserve F2 frequency, signed gain, waveform and all other fields.
        Missing cfg or unresolved phase returns no proposal. A valid proposal
        requires the Run/Load parameter capture to resolve its concrete library
        key; it never reads current pulse values. No writes or atomic pair Apply.
        """
        snapshot = req.run_result.cfg
        phase = req.analyze_result.relative_phase_deg
        if snapshot is None or phase is None or not np.isfinite(phase):
            return []
        parameters = req.ctx.parameters
        if parameters is None:
            raise ValueError(
                "Constructive phase writeback requires captured parameters"
            )
        target = f"{snapshot.roles.coupler}.pulses.cz_f2"
        key = parameters.library_key(target)
        return [
            module_writeback(
                snapshot.drive.f2,
                target=target,
                description="Constructive F2 charge pulse, source F1 phase reference",
                field_updates=(("phase", float(snapshot.drive.f1.phase) + phase),),
            ),
            ParameterWriteback(
                target_name=f"{snapshot.roles.coupler}.module.cz_f2",
                description="F2 CZ pulse slot, independent Apply row",
                proposed_value=key,
            ),
        ]
