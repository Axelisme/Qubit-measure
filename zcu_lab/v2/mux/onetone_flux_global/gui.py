"""Run and pick one participant of the shared-coil onetone flux map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadataContent
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.components.qubit import QubitBaseSchema
from zcu_lab.v2.support.flux import (
    device_flux_points,
    snapshot_numeric_values,
)
from zcu_lab.v2.support.measure.mux_results import captured_qubits
from zcu_lab.v2.support.measure.mux_save_names import mux_save_stem
from zcu_lab.v2.support.measure.writeback_helpers import fluxdep_writeback_items
from zcu_lab.v2.support.mux_flux import (
    MuxFluxResult,
    validate_flux_result,
)

from .core import OnetoneFluxGlobalCfg, OnetoneFluxGlobalExp
from .form import make_form


@dataclass
class MuxFluxPickParams:
    """Select one acquired original tone and optional initial DC line positions.

    tone is the full readout index, not the participant row; an unacquired index
    raises ValueError at analysis start. flux_half and flux_int are finite
    half/integer-flux DC values in the scanned device's native unit (cfg.flux_unit
    labels A, V or native without conversion). Each None independently uses
    that tone's Run-captured calibration, then the shared picker default if
    missing. Explicit values override only this analysis, never the capture.
    The picker folds the pair near the scan center using its existing period
    rule; nonfinite positions raise TypeError/ValueError. done publishes one
    tone/Q result and its proposals; it does not accept writeback.
    """

    tone: int = field(
        default=0,
        metadata={"description": "Original tone index of the participant to calibrate"},
    )
    flux_half: float | None = field(
        default=None,
        metadata={"description": "Initial half-flux DC value; None uses Run capture"},
    )
    flux_int: float | None = field(
        default=None,
        metadata={
            "description": "Initial integer-flux DC value; None uses Run capture"
        },
    )


@dataclass
class MuxFluxPickResult(FluxPickResult):
    """Accepted native-unit half/int/period and this Run's original Q identity."""

    tone: int
    qubit: str


def get_analyze_params(
    result: RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult], ctx: SessionEnv
) -> MuxFluxPickParams:
    """Start with the first acquired original index, including reordered subsets."""
    del ctx
    validate_flux_result(result.result)
    return MuxFluxPickParams(tone=int(result.result.tone_indices[0]))


def make_interactive_plugin(
    req: AnalyzeRequest[
        RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult], MuxFluxPickParams
    ],
    *,
    plots: Plots,
) -> FluxPickPlugin[MuxFluxPickResult]:
    """Pick half/integer lines on one complete-partial map from this Run.

    tone must name an acquired original index. Completed flux rows survive
    Stop; unfinished rows are excluded. Historical cfg is required for Q
    identity. Missing capture permits picking but not parameter proposals.
    Frequencies are converted from core MHz to the shared picker\'s GHz.
    The picker starts in magnitude mode. Each analysis
    returns one Q's lines; repeat Analyze for other acquired participants.
    """
    data, cfg = req.run_result.result, req.run_result.cfg
    validate_flux_result(data)
    if cfg is None:
        raise ValueError("Mux flux picking requires the saved full cfg for Q identity")
    names = tuple(t.name for t in cfg.tones)
    captured_qubits(req.parameters, names)
    tone = req.analyze_params.tone
    if isinstance(tone, bool) or tone not in data.tone_indices:
        raise ValueError(f"tone={tone!r} is not an acquired original tone index")
    row = list(data.tone_indices).index(tone)
    complete = np.isfinite(data.signals[row]).all(axis=1)
    inputs = FluxPickInputs(
        data.signals[row, complete],
        data.flux_values[row, complete],
        data.frequencies[row] * 1e-3,  # The shared picker displays GHz.
    )
    half, integer = _initial_lines(req, tone)
    if req.analyze_params.flux_half is not None:
        half = req.analyze_params.flux_half
    if req.analyze_params.flux_int is not None:
        integer = req.analyze_params.flux_int

    def build(
        inputs: FluxPickInputs, state: FluxPickState, plots: Plots
    ) -> MuxFluxPickResult:
        picked = render_flux_pick(inputs, state, plots)
        return MuxFluxPickResult(
            picked.flx_half, picked.flx_int, picked.flx_period, tone, names[tone]
        )

    return make_flux_pick_plugin(
        inputs,
        flux_half=half,
        flux_int=integer,
        force_magnitude=True,
        plots=plots,
        result_builder=build,
    )


def _initial_lines(
    req: AnalyzeRequest[
        RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult], MuxFluxPickParams
    ],
    tone: int,
) -> tuple[float | None, float | None]:
    parameters, cfg = req.parameters, req.run_result.cfg
    if parameters is None or cfg is None:
        return None, None
    name = cfg.tones[tone].name
    component = parameters.document().components.get(name)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{name!r}: flux picking requires a qubit component")
    device = cfg.flux_dev
    validate_device_name(device)
    if component.flux.offset is None or device not in component.flux.period:
        return None, None
    metadata = RunMetadataContent(req.run_result.metadata)
    snapshot = metadata.device_snapshot
    values = snapshot_numeric_values(snapshot, component.flux.period)
    points = device_flux_points(component.flux, device, values)
    return points.half, points.integer


def get_writeback_items(
    req: WritebackRequest[
        RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult], MuxFluxPickResult
    ],
) -> Sequence[WritebackItem]:
    """Propose only this analysis's Q calibration using the Run capture.

    Missing capture or cfg yields no proposals. Identity mismatches raise
    instead of guessing a target. Propose the selected signed period in
    qubit.flux.period, qubit.flux.offset and a raw-fit record, using the actual
    cfg.flux_dev name. Conversion and record evidence use only Run parameters
    and the device snapshot.
    """
    cfg, picked = req.run_result.cfg, req.analyze_result
    if cfg is None:
        return []
    qubits = captured_qubits(req.ctx.parameters, tuple(t.name for t in cfg.tones))
    if qubits is None:
        return []
    if (
        picked.tone not in req.run_result.result.tone_indices
        or qubits[picked.tone] != picked.qubit
    ):
        raise ValueError("Picked tone/Q identity does not match the Run capture")
    identity = ElementIdentity(index=picked.tone, name=picked.qubit)
    return fluxdep_writeback_items(
        req,
        qubit=picked.qubit,
        device_name=cfg.flux_dev,
        element=identity,
    )


def run(
    cfg: OnetoneFluxGlobalCfg, *, context: RunContext
) -> RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult]:
    """Acquire the shared-coil map with context and retain its actual cfg."""
    return RunRecord(cfg=cfg, result=OnetoneFluxGlobalExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=replace(OnetoneFluxGlobalExp.AXES_SPEC, save_stem=mux_save_stem),
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=make_interactive_plugin,
        frontend=make_flux_pick_frontend,
        options=get_analyze_params,
        writeback=get_writeback_items,
    ),
)
