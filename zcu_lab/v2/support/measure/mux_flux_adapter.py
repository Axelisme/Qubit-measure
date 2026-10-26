"""One-participant flux picking and fixed-capture calibration proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.mux.onetone.core import MuxFrequencyOffsetCfg
from zcu_lab.v2.support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2.support.measure.mux_bindings import require_mux_group
from zcu_lab.v2.support.measure.mux_results import captured_qubits
from zcu_lab.v2.support.measure.writeback_helpers import fluxdep_writeback_items
from zcu_lab.v2.support.mux_flux import (
    FluxFrequencyCfg,
    GlobalFluxCfg,
    LocalFluxCfg,
    MuxFluxResult,
    validate_flux_result,
)


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


class MuxFluxPickSummary(BaseModel):
    """Picked native DC half/integer/period values and captured original tone/Q identity."""

    tone: int
    qubit: str
    flx_half: float | None
    flx_int: float | None
    flx_period: float | None


@dataclass
class MuxFluxPickResult(FluxPickResult):
    """Accepted native-unit half/int/period and this Run's original Q identity."""

    tone: int
    qubit: str


type MuxFluxCfg = (
    GlobalFluxCfg[MuxFrequencyOffsetCfg]
    | LocalFluxCfg[MuxFrequencyOffsetCfg]
    | GlobalFluxCfg[FluxFrequencyCfg]
    | LocalFluxCfg[FluxFrequencyCfg]
)


def get_analyze_params[C: MuxFluxCfg](
    result: RunRecord[C, MuxFluxResult], ctx: SessionEnv
) -> MuxFluxPickParams:
    """Start with the first acquired original index, including reordered subsets."""
    del ctx
    validate_flux_result(result.result)
    return MuxFluxPickParams(tone=int(result.result.tone_indices[0]))


def make_interactive_plugin[C: MuxFluxCfg](
    req: AnalyzeRequest[RunRecord[C, MuxFluxResult], MuxFluxPickParams],
    *,
    plots: Plots,
    force_magnitude: bool = True,
) -> FluxPickPlugin[MuxFluxPickResult]:
    """Pick half/integer lines on one complete-partial map from this Run.

    tone must name an acquired original index. Completed flux rows survive
    Stop; unfinished rows are excluded. Historical cfg is required for Q
    identity. Missing capture permits picking but not parameter proposals.
    Frequencies are converted from core MHz to the shared picker\'s GHz.
    Onetone starts in magnitude, twotone in phase-aware mode. Each analysis
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
        force_magnitude=force_magnitude,
        plots=plots,
        result_builder=build,
    )


def _initial_lines[C: MuxFluxCfg](
    req: AnalyzeRequest[RunRecord[C, MuxFluxResult], MuxFluxPickParams],
    tone: int,
) -> tuple[float | None, float | None]:
    parameters, cfg = req.parameters, req.run_result.cfg
    if parameters is None or cfg is None:
        return None, None
    if isinstance(cfg, GlobalFluxCfg):
        group = require_mux_group(parameters)
        calibration = group.coil_flux.get(cfg.tones[tone].name)
        return (
            (None, None) if calibration is None else (calibration.half, calibration.int)
        )
    name = cfg.tones[tone].name
    return optional_parameter_float(
        parameters, f"{name}.flux_half"
    ), optional_parameter_float(parameters, f"{name}.flux_int")


def get_writeback_items[C: MuxFluxCfg](
    req: WritebackRequest[RunRecord[C, MuxFluxResult], MuxFluxPickResult],
) -> Sequence[WritebackItem]:
    """Propose only this analysis's Q calibration using the Run capture.

    Missing capture or cfg yields no proposals. Identity mismatches raise
    instead of guessing a target. Both scan types propose only the selected
    device entry in qubit.flux and an ordinary raw-fit record. Global uses the
    actual cfg.flux_dev name, not a reserved coil alias. Local rejects another
    simultaneously scanned participant device if it contributes to this Q.
    Conversion and record evidence use only Run parameters and device snapshot.
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
    if isinstance(cfg, GlobalFluxCfg):
        return fluxdep_writeback_items(
            req,
            qubit=picked.qubit,
            device_name=cfg.flux_dev,
            swept_devices=(cfg.flux_dev,),
            element=identity,
        )
    return fluxdep_writeback_items(
        req,
        qubit=picked.qubit,
        device_name=cfg.tones[picked.tone].flux_dev,
        swept_devices=tuple(cfg.tones[index].flux_dev for index in cfg.participants),
        element=identity,
    )
