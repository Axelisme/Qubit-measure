"""Measure adapter for fixed-depth interleaved RB optimization."""

from collections.abc import Sequence
from pathlib import Path

from zcu_tools.cfg_model import PulseSpec
from zcu_tools.core.waveform_format import hash_samples
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ArbAssetWriteback,
    ModuleWriteback,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import (
    CZOptimizeAnalysis,
    CZOptimizeCfg,
    CZOptimizeExp,
    CZOptimizeResult,
    CZOptimizeUnavailable,
)
from .form import make_form

type _Run = RunRecord[CZOptimizeCfg, CZOptimizeResult]


def _analyze(
    req: AnalyzeRequest[_Run, None], *, plots: Plots
) -> CZOptimizeAnalysis | CZOptimizeUnavailable:
    """Compare measured evaluations from the selected Run or Load record."""
    return CZOptimizeExp().analyze(req.run_result, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[_Run, CZOptimizeAnalysis | CZOptimizeUnavailable],
) -> Sequence[WritebackItem]:
    cfg, best = req.run_result.cfg, req.analyze_result
    if isinstance(best, CZOptimizeUnavailable):
        return []
    if cfg is None:
        raise ValueError(
            "CZ optimization writeback requires the saved starting cfg; see Load diagnostics"
        )
    parameters = req.ctx.parameters
    if parameters is None:
        raise ValueError("CZ optimization writeback requires a Run/Load capture")

    samples = best.render_best() if best.best_evaluation else None
    key = f"cz_opt_{hash_samples(samples)[:12]}" if samples is not None else None
    selected = best.start_target is not None and best.best_target > best.start_target
    note = None
    if best.start_target is None:
        note = (
            "The starting survival is unavailable; no improvement can be established."
        )
    elif not selected:
        note = "The best measured survival does not exceed the starting survival."

    # ADR-0008: one draft can publish the asset and apply the pulses that use it.
    items: list[WritebackItem] = []
    if samples is not None and key is not None:
        items.append(
            ArbAssetWriteback(
                target_name=key,
                description="Best CZ envelope for both drive lines",
                samples=samples,
                selected=selected,
                note=note,
            )
        )
    for index, (position, pulse) in enumerate(
        (("qubit1", cfg.drive.qubit1), ("qubit2", cfg.drive.qubit2))
    ):
        label = position.capitalize()
        changes = {}
        if key is not None and best.best_raw_gains is not None:
            changes = {
                "gain": best.best_raw_gains[index],
                "waveform": {"style": "arb", "data": key},
            }
        # A saved native arb adds a host path, which is not part of a persistent Spec.
        proposal = PulseSpec.model_validate(
            {**pulse.to_dict(), **changes}, extra="ignore"
        )
        # Pulse library keys: $coupler.pulses.cz_qubit1 and $coupler.pulses.cz_qubit2.
        target = f"$coupler.pulses.cz_{position}"
        items.extend(
            (
                ModuleWriteback(
                    target_name=target,
                    description=f"Selected CZ {label} pulse",
                    proposal=proposal,
                    selected=selected,
                    note=note,
                ),
                ParameterWriteback(
                    # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
                    target_name=f"$coupler.module.cz_{position}",
                    description=f"CZ {label} pulse slot",
                    proposed_value=parameters.library_key(target),
                    selected=selected,
                    note=note,
                ),
            )
        )
    angle1, angle2 = (
        best.corrections_deg
        if best.best_evaluation
        else (
            cfg.phase_pair.correction_qubit1_deg,
            cfg.phase_pair.correction_qubit2_deg,
        )
    )
    items.append(
        ParameterWriteback(
            # Keys: $coupler.ext.cz_phase.qubit1, $coupler.ext.cz_phase.qubit2,
            # $coupler.ext.cz_phase.correction_qubit1_deg,
            # $coupler.ext.cz_phase.correction_qubit2_deg.
            target_name="$coupler.ext.cz_phase",
            description="Best logical Z corrections for Qubit1 and Qubit2 (deg)",
            proposed_value={
                "qubit1": parameters.resolve_path("$qubit1"),
                "qubit2": parameters.resolve_path("$qubit2"),
                "correction_qubit1_deg": angle1,
                "correction_qubit2_deg": angle2,
            },
            selected=selected,
            note=note,
        )
    )
    return items


def run(cfg: CZOptimizeCfg, *, context: RunContext) -> _Run:
    """Run with captured calibration and retain the original cfg, not a candidate."""
    return RunRecord(cfg=cfg, result=CZOptimizeExp().run(cfg, context=context))


def _summary_lines(source: _Run) -> tuple[str, ...]:
    cfg = source.cfg
    duration = f"Starting CZ duration {source.result.start_duration_s * 1e9:.6g} ns"
    return (duration,) if cfg is None else (f"Reps×Rounds {cfg.reps}×1", duration)


DEFINITION = Adapter(
    spec=CZOptimizeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, writeback=_get_writeback_items, summary_lines=_summary_lines
    ),
)
