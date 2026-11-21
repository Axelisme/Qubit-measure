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
from zcu_tools.plotting.plots import Plots

from ...support.measure.writeback_helpers import module_writeback
from .core import (
    ConditionalPhaseAnalysis,
    ConditionalPhaseCfg,
    ConditionalPhaseExp,
    ConditionalPhaseResult,
)
from .form import make_form

ConditionalPhaseRun: TypeAlias = RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult]


def _analyze(
    req: AnalyzeRequest[ConditionalPhaseRun, None], *, plots: Plots
) -> ConditionalPhaseAnalysis:
    """Use only source cfg/counts; never fill missing quadratures from current inputs."""
    return ConditionalPhaseExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[ConditionalPhaseRun, ConditionalPhaseAnalysis],
) -> Sequence[WritebackItem]:
    """Propose two ordinary source pulses with only common frequency replaced.

    A unique fitted per-gate crossing, saved cfg and fixed Run/Load capture are required.
    Preserve signed gain ratio, phases, waveform and timing. Module roots and
    independent slot rows target the captured coupler. FIT has no write effects;
    preview and selected Apply belong to the operator, not a paired transaction.
    """
    cfg, parameters = req.run_result.cfg, req.ctx.parameters
    frequency = req.analyze_result.frequency_mhz
    if parameters is None:
        raise ValueError("Conditional phase writeback requires captured parameters")
    if cfg is None or frequency is None:
        return []
    coupler = parameters.resolve_path("$coupler")
    items: list[WritebackItem] = []
    for role, pulse in (("qubit1", cfg.drive.qubit1), ("qubit2", cfg.drive.qubit2)):
        # Pulse library keys: $coupler.pulses.cz_qubit1 and $coupler.pulses.cz_qubit2.
        root = f"{coupler}.pulses.cz_{role}"
        items.append(
            module_writeback(
                pulse,
                target=root,
                description=f"CZ {role.capitalize()} source pulse with fitted common frequency (MHz)",
                field_updates=(("freq", frequency),),
            )
        )
        items.append(
            ParameterWriteback(
                # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
                target_name=f"{coupler}.module.cz_{role}",
                description=f"Select captured coupler CZ {role.capitalize()} pulse",
                proposed_value=parameters.library_key(root),
            )
        )
    return items


def run(
    cfg: ConditionalPhaseCfg, *, context: RunContext
) -> RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ConditionalPhaseExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=ConditionalPhaseExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
