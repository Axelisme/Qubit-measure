"""Mux t2ramsey source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadataContent
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    RecordWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.flux import record_flux_frame
from zcu_lab.v2.support.measure.mux_results import captured_qubits
from zcu_lab.v2.support.measure.mux_save_names import mux_save_stem
from zcu_lab.v2.support.measure.record_frame import captured_qubit_flux
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_coherence import MuxCoherenceResult
from zcu_lab.v2.support.mux_coherence_analysis import MuxCoherenceFit

from .core import MuxT2RamseyCfg, MuxT2RamseyExp, T2RamseyAnalyzeOptions
from .form import make_form

MuxT2RamseyRunResult: TypeAlias = RunRecord[MuxT2RamseyCfg, MuxCoherenceResult]


@dataclass(frozen=True)
class MuxT2RamseyAnalyzeResult:
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxCoherenceFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None


def _frequency_review_note(outcome: MuxCoherenceFit, t2_note: str | None) -> str | None:
    """Add sampling and frequency-uncertainty review to the T2 gate.

    Require adequate delay sampling and finite nonnegative frequency error / |MHz
    fringe rate| <=0.5. Zero frequency has no relative error and needs review.
    Adequate sampling still cannot exclude unknown aliases. Mux Ramsey mainly
    estimates the decay trend, not precision detuning; users may inspect/override.
    """
    reasons = [] if t2_note is None else [t2_note]
    sampling = outcome.sampling_quality
    if sampling is None:
        reasons.append("fringe sampling unavailable")
    elif sampling.status != "adequate":
        reasons.append(
            f"fringe sampling {sampling.status}: {sampling.samples_per_cycle} "
            f"samples/cycle, minimum 10; reason={sampling.reason}"
        )
    if not math.isfinite(outcome.detune) or outcome.detune == 0:
        reasons.append("frequency relative error unavailable for zero/nonfinite rate")
    elif not math.isfinite(outcome.detune_err) or outcome.detune_err < 0:
        reasons.append("frequency error must be finite and nonnegative")
    elif outcome.detune_err / abs(outcome.detune) > 0.5:
        reasons.append(
            f"frequency relative error={outcome.detune_err / abs(outcome.detune):.4g} >0.5"
        )
    reasons.extend(
        f"frequency {issue.path}: {issue.reason}"
        for issue in outcome.quality.invalid
        if issue.path == "relative_parameter_errors.freq"
    )
    return "; ".join(reasons) if reasons else None


def coherence_review_note(
    t2: float, t2_err: float, quality: FitQuality | None
) -> str | None:
    """Explain why this tone's T2 fit needs manual selection.

    Times/errors are in us. Initial selection requires positive finite T2,
    nonnegative finite error/T2 <=0.5, r2 >=0.9 and normalized residual RMS <=0.1.
    Missing/nonfinite required metrics or their invalid issues require review.
    Unrelated fixed-zero parameter issues do not reject the result. None means
    these review gates passed, not that a calibration is physically established.
    """
    reasons = _coherence_fit_review_reasons(quality)
    if not math.isfinite(t2) or t2 <= 0:
        reasons.append("T2 must be finite and positive")
    elif not math.isfinite(t2_err) or t2_err < 0:
        reasons.append("T2 error must be finite and nonnegative")
    elif t2_err / t2 > 0.5:
        reasons.append(f"T2 relative error={t2_err / t2:.4g} >0.5")
    return "Review required: " + "; ".join(reasons) if reasons else None


def _coherence_fit_review_reasons(quality: FitQuality | None) -> list[str]:
    if quality is None:
        return ["fit quality unavailable"]
    reasons = [
        f"{issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.decay_time"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"normalized RMS={rms:.4g} >0.1")
    return reasons


def _MuxT2Ramsey_options(
    result: MuxT2RamseyRunResult, ctx: SessionEnv
) -> T2RamseyAnalyzeOptions:
    """Seed the analysis controls without changing the core defaults."""
    del result, ctx
    return T2RamseyAnalyzeOptions(fit_fringe=True, fit_phase=True)


def _analyze(
    req: AnalyzeRequest[MuxT2RamseyRunResult, T2RamseyAnalyzeOptions], *, plots: Plots
) -> MuxT2RamseyAnalyzeResult:
    """Analyze the explicit Run/Load source, retaining each original-index outcome."""
    outcomes = MuxT2RamseyExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    snapshot = req.run_result.cfg
    return MuxT2RamseyAnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxT2RamseyRunResult, MuxT2RamseyAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose only complete, successful tones using the same Run/Load capture.

    Missing capture returns no items. Captured bindings must match any source
    cfg names/count; errors propagate. Partial/nonfinite rows and failed fits
    produce no calibration items. Pulse proposals copy source cfg only, never
    current library values. T2/record require the coherence review gate;
    frequency additionally requires adequate sampling and relative error <=0.5.
    Failed gates retain unselected candidates with notes for manual review.
    This method neither writes nor applies proposals.
    """
    metadata = RunMetadataContent(req.run_result.metadata)
    devices = (
        None
        if metadata.device_snapshot is None
        else deepcopy(dict(metadata.device_snapshot))
    )
    if devices is None:
        logging.warning("no device snapshot")
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
        qubit = qubits[outcome.index]
        t2_note = coherence_review_note(outcome.t2, outcome.t2_err, outcome.quality)
        row_items: list[WritebackItem] = []
        row_items.extend(
            [
                ParameterWriteback(
                    target_name=f"{qubit}.t2r",
                    description="t2r coherence time (µs)",
                    proposed_value=outcome.t2,
                    selected=t2_note is None,
                    note=t2_note,
                ),
            ]
        )
        if devices is not None:
            frame = record_flux_frame(
                captured_qubit_flux(req.ctx.parameters, qubit), devices
            )
            row_items.append(
                RecordWriteback(
                    target_name=qubit,
                    description="t2r record at captured bias",
                    record={
                        "devices": {
                            name: deepcopy(info) for name, info in devices.items()
                        },
                        "flux_frame": frame,
                        "experiment": metadata.experiment,
                        "t2r_us": outcome.t2,
                        "t2r_err_us": outcome.t2_err,
                    },
                    selected=t2_note is None,
                    note=t2_note,
                )
            )
        data = req.run_result.result
        if (
            snapshot is not None
            and outcome.fit_fringe
            and np.isfinite(data.actual_detunes[row])
            and np.isfinite(data.source_frequencies[row])
            and np.isfinite(outcome.detune)
        ):
            frequency = (
                data.source_frequencies[row] + data.actual_detunes[row] - outcome.detune
            )
            frequency_note = _frequency_review_note(outcome, t2_note)
            row_items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.freq",
                    description="Qubit frequency from sourced Ramsey fringe (MHz)",
                    proposed_value=float(frequency),
                    selected=frequency_note is None,
                    note=frequency_note,
                )
            )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def run(
    cfg: MuxT2RamseyCfg, *, context: RunContext
) -> RunRecord[MuxT2RamseyCfg, MuxCoherenceResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxT2RamseyExp().run(cfg, context=context))


def _summary_lines(source: MuxT2RamseyRunResult) -> tuple[str, ...]:
    cfg, result = source.cfg, source.result
    lines = [] if cfg is None else [f"Reps×Rounds {cfg.reps}×{cfg.rounds}"]
    for row, index in enumerate(result.tone_indices):
        tone = f"Tone {index}" if cfg is None else cfg.tones[int(index)].name
        detune = result.actual_detunes[row]
        if np.isfinite(detune):
            lines.append(f"{tone} · Applied detune = {detune:.6f} MHz")
        carrier = result.source_frequencies[row]
        if np.isfinite(carrier):
            lines.append(f"{tone} · π/2 pulse source frequency = {carrier:.6f} MHz")
    return tuple(lines)


DEFINITION = Adapter(
    spec=replace(MuxT2RamseyExp.AXES_SPEC, save_stem=mux_save_stem),
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=_MuxT2Ramsey_options,
        writeback=_get_writeback_items,
        summary_lines=_summary_lines,
    ),
)
