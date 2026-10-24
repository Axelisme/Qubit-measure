"""Measure adapter for conditional phase and common CZ frequency."""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from ..._support.measure.writeback_helpers import module_writeback
from .analysis import ConditionalPhaseAnalysis
from .core import ConditionalPhaseCfg, ConditionalPhaseExp, ConditionalPhaseResult
from .form import make_form

ConditionalPhaseRun: TypeAlias = RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult]


class ConditionalPhaseAdapter(
    BaseAdapter[ConditionalPhaseCfg, ConditionalPhaseRun, ConditionalPhaseAnalysis]
):
    """Fixed-capture Form, full raw acquisition, diagnostic analysis and restore."""

    exp_cls = ConditionalPhaseExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ConditionalPhaseCfg]:
        """Resolve role/tone identities and editable pulses from one Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: ConditionalPhaseCfg, *, context: RunContext
    ) -> ConditionalPhaseRun:
        """Keep original cfg with raw; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(
            cfg=cfg, result=ConditionalPhaseExp().run(cfg, context=context)
        )

    def analyze(
        self, req: AnalyzeRequest[ConditionalPhaseRun, None], *, plots: Plots
    ) -> ConditionalPhaseAnalysis:
        """Use only source cfg/raw; never fill missing quadratures from current inputs."""
        return ConditionalPhaseExp().analyze(req.run_result, None, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[ConditionalPhaseRun, ConditionalPhaseAnalysis]
    ) -> Sequence[WritebackItem]:
        """Propose two ordinary source pulses with only common frequency replaced.

        A unique n=1 crossing, saved cfg and fixed Run/Load capture are required.
        Preserve signed gain ratio, phases, waveform and timing. Module roots and
        independent slot rows target the captured coupler. FIT has no write effects;
        preview and selected Apply belong to the operator, not a paired transaction.
        """
        cfg, parameters = req.run_result.cfg, req.ctx.parameters
        frequency = req.analyze_result.frequency_mhz
        if cfg is None or parameters is None or frequency is None:
            return []
        items: list[WritebackItem] = []
        for role, pulse in (("f1", cfg.drive.f1), ("f2", cfg.drive.f2)):
            root = f"$coupler.pulses.cz_{role}"
            items.append(
                module_writeback(
                    pulse,
                    target=root,
                    description=f"CZ {role.upper()} source pulse with n=1 common frequency (MHz)",
                    field_updates=(("freq", frequency),),
                )
            )
            items.append(
                ParameterWriteback(
                    target_name=f"$coupler.module.cz_{role}",
                    description=f"Select captured coupler CZ {role.upper()} pulse",
                    proposed_value=parameters.resolve_path(root),
                )
            )
        return items
