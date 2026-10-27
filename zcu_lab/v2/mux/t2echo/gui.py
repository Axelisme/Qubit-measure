"""Mux t2echo source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
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
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t2echo.gui import coherence_review_note

from .core import MuxT2EchoCfg, MuxT2EchoExp
from .form import make_form

MuxT2EchoRunResult: TypeAlias = RunRecord[MuxT2EchoCfg, MuxCoherenceResult]


@dataclass(frozen=True)
class MuxT2EchoAnalyzeResult:
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxCoherenceFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None


def _MuxT2Echo_options(
    result: MuxT2EchoRunResult, ctx: SessionEnv
) -> T2EchoAnalyzeOptions:
    """Seed the analysis controls without changing the core defaults."""
    del result, ctx
    return T2EchoAnalyzeOptions(fit_method="decay", fit_phase=False)


def _analyze(
    req: AnalyzeRequest[MuxT2EchoRunResult, T2EchoAnalyzeOptions], *, plots: Plots
) -> MuxT2EchoAnalyzeResult:
    """Analyze the explicit Run/Load source, retaining each original-index outcome."""
    outcomes = MuxT2EchoExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    snapshot = req.run_result.cfg
    return MuxT2EchoAnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxT2EchoRunResult, MuxT2EchoAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose only complete, successful tones using the same Run/Load capture.

    Missing capture returns no items. Captured bindings must match any source
    cfg names/count; errors propagate. Partial/nonfinite rows and failed fits
    produce no calibration items. Pulse proposals copy source cfg only, never
    current library values. T2 and its record share the coherence review gate:
    r2 >=0.9, normalized RMS <=0.1 and error/T2 <=0.5. Poor or unavailable
    quality leaves candidates unselected with reasons; users may override.
    This method neither writes nor applies proposals.
    """
    metadata = req.run_result.metadata
    devices = (
        None
        if metadata is None or metadata.device_snapshot is None
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
        note = coherence_review_note(outcome.t2, outcome.t2_err, outcome.quality)
        row_items: list[WritebackItem] = []
        row_items.extend(
            [
                ParameterWriteback(
                    target_name=f"{qubit}.t2e",
                    description="t2e coherence time (us)",
                    proposed_value=outcome.t2,
                    selected=note is None,
                    note=note,
                ),
            ]
        )
        if metadata is not None and devices is not None:
            frame = record_flux_frame(
                captured_qubit_flux(req.ctx.parameters, qubit), devices
            )
            row_items.append(
                RecordWriteback(
                    target_name=qubit,
                    description="t2e record at captured bias",
                    record={
                        "devices": {
                            name: deepcopy(info) for name, info in devices.items()
                        },
                        "flux_frame": frame,
                        "run": {
                            "run_id": metadata.run_id,
                            "experiment": metadata.experiment,
                        },
                        "t2e_us": outcome.t2,
                        "t2e_err_us": outcome.t2_err,
                    },
                    selected=note is None,
                    note=note,
                )
            )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def run(
    cfg: MuxT2EchoCfg, *, context: RunContext
) -> RunRecord[MuxT2EchoCfg, MuxCoherenceResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxT2EchoExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=replace(MuxT2EchoExp.AXES_SPEC, save_stem=mux_save_stem),
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=_MuxT2Echo_options, writeback=_get_writeback_items
    ),
)
