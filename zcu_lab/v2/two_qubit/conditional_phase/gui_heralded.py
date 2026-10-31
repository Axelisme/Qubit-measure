"""Measure adapter for conditional phase and common CZ frequency."""

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from ...support.measure.writeback_helpers import module_writeback
from .analysis import ConditionalPhaseAnalysis
from .core_heralded import (
    ConditionalPhaseHeraldedCfg,
    ConditionalPhaseHeraldedExp,
    ConditionalPhaseHeraldedResult,
)
from .form import phase_form

ConditionalPhaseHeraldedRun: TypeAlias = RunRecord[ConditionalPhaseHeraldedCfg, ConditionalPhaseHeraldedResult]


def make_form(ctx: FormContext) -> Form[ConditionalPhaseHeraldedCfg]:
    """Capture roles and seed the independent heralded experiment."""
    return phase_form(ctx, ConditionalPhaseHeraldedCfg)


def _analyze(
    req: AnalyzeRequest[ConditionalPhaseHeraldedRun, None], *, plots: Plots
) -> ConditionalPhaseAnalysis:
    """Use only source cfg/counts; never fill missing quadratures from current inputs."""
    return ConditionalPhaseHeraldedExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[ConditionalPhaseHeraldedRun, ConditionalPhaseAnalysis],
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


def run(
    cfg: ConditionalPhaseHeraldedCfg, *, context: RunContext
) -> RunRecord[ConditionalPhaseHeraldedCfg, ConditionalPhaseHeraldedResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ConditionalPhaseHeraldedExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=ConditionalPhaseHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
