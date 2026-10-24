"""Measure adapter for conditional phase and common CZ frequency."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
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
from .analysis import ConditionalPhaseAnalysis
from .core import ConditionalPhaseCfg, ConditionalPhaseExp, ConditionalPhaseResult
from .form import make_form

ConditionalPhaseRun: TypeAlias = RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult]


@dataclass(frozen=True)
class ConditionalPhaseAnalyzeResult:
    """JSON-safe ensemble phase summary; diagnostic plots remain named figures."""

    analysis: ConditionalPhaseAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Report projection counts, branch phases and reasons without nonfinite JSON."""
        return {
            "branches": [
                {
                    "spectator": ("I", "Xπ")[b],
                    "gate_count": self.analysis.gate_counts[n],
                    "points": [
                        {
                            "reason": p.reason,
                            "frequency_mhz": self.analysis.frequencies_mhz[f],
                            "contrast": p.contrast if np.isfinite(p.contrast) else None,
                            "logical_phase_deg": p.logical_phase_deg,
                        }
                        for f, p in enumerate(points)
                    ],
                }
                for b, trains in enumerate(self.analysis.phases)
                for n, points in enumerate(trains)
            ],
            "conditional_phase_deg": [
                [float(p) if np.isfinite(p) else None for p in row]
                for row in self.analysis.conditional_deg
            ],
            "acquired_shots": self.analysis.acquired_shots.tolist(),
            "frequency_mhz": self.analysis.frequency_mhz,
            "frequency_reason": self.analysis.frequency_reason,
            "crossings": [
                {
                    "gate_count": n,
                    "frequencies_mhz": list(c.frequencies_mhz),
                    "reason": c.reason,
                }
                for n, c in zip(
                    self.analysis.gate_counts, self.analysis.crossings, strict=True
                )
            ],
            "frequency_method": "n=1 adjacent unwrap and piecewise-linear interpolation; higher n diagnostic only. No fitted frequency uncertainty model.",
            "interpretation": "Thermal nominal I/Xπ branches; GE-normalized ensemble phase. No heralding, shot exclusion, matrix inversion or physical fidelity claim.",
        }


def _analyze(
    req: AnalyzeRequest[ConditionalPhaseRun, None], *, plots: Plots
) -> ConditionalPhaseAnalyzeResult:
    """Use only source cfg/raw; never fill missing quadratures from current inputs."""
    return ConditionalPhaseAnalyzeResult(
        ConditionalPhaseExp().analyze(req.run_result, None, plots=plots)
    )


def _get_writeback_items(
    req: WritebackRequest[ConditionalPhaseRun, ConditionalPhaseAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose two ordinary source pulses with only common frequency replaced.

    A unique n=1 crossing, saved cfg and fixed Run/Load capture are required.
    Preserve signed gain ratio, phases, waveform and timing. Module roots and
    independent slot rows target the captured coupler. FIT has no write effects;
    preview and selected Apply belong to the operator, not a paired transaction.
    """
    cfg, parameters = req.run_result.cfg, req.ctx.parameters
    frequency = req.analyze_result.analysis.frequency_mhz
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
