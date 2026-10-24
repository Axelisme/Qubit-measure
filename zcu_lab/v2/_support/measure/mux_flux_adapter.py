"""One-participant flux picking and fixed-capture calibration proposals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from pydantic import BaseModel
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.core.element import ElementIdentity
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
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2._support.measure.mux_bindings import require_mux_group
from zcu_lab.v2._support.measure.mux_results import captured_qubits
from zcu_lab.v2._support.mux_flux import (
    FluxFrequencyCfg,
    GlobalFluxCfg,
    LocalFluxCfg,
    MuxFluxResult,
    run_flux_scan,
    validate_flux_result,
)
from zcu_lab.v2.mux.onetone.core import MuxFrequencyOffsetCfg


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


class MuxFluxAdapter[
    C: GlobalFluxCfg[MuxFrequencyOffsetCfg]
    | LocalFluxCfg[MuxFrequencyOffsetCfg]
    | GlobalFluxCfg[FluxFrequencyCfg]
    | LocalFluxCfg[FluxFrequencyCfg]
](BaseAdapter[C, RunRecord[C, MuxFluxResult], MuxFluxPickResult, MuxFluxPickParams]):
    """Reuse the single-map picker for one acquired participant at a time.

    Concrete adapters supply the form, experiment class and guide. Every
    analysis retains its original tone and qubit name in the returned summary.
    Destinations come only from the Run/Load capture, never current aliases.
    """

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=False
    )
    force_magnitude: ClassVar[bool] = True

    def run(
        self, req: RunRequest, cfg: C, *, context: RunContext
    ) -> RunRecord[C, MuxFluxResult]:
        """Acquire the captured cfg's ordered participants and retain complete settings."""
        del req
        return RunRecord(cfg, run_flux_scan(cfg, context=context))

    def get_analyze_params(
        self, result: RunRecord[C, MuxFluxResult], ctx: SessionEnv
    ) -> MuxFluxPickParams:
        """Start with the first acquired original index, including reordered subsets."""
        del ctx
        validate_flux_result(result.result)
        return MuxFluxPickParams(tone=int(result.result.tone_indices[0]))

    def make_interactive_plugin(
        self,
        req: AnalyzeRequest[RunRecord[C, MuxFluxResult], MuxFluxPickParams],
        *,
        plots: Plots,
    ) -> FluxPickPlugin:
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
            raise ValueError(
                "Mux flux picking requires the saved full cfg for Q identity"
            )
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
        half, integer = self._initial_lines(req, tone)
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
            force_magnitude=self.force_magnitude,
            plots=plots,
            result_builder=build,
        )

    def _initial_lines(
        self,
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
                (None, None)
                if calibration is None
                else (calibration.half, calibration.int)
            )
        name = cfg.tones[tone].name
        return optional_parameter_float(
            parameters, f"{name}.flux_half"
        ), optional_parameter_float(parameters, f"{name}.flux_int")

    def make_interactive_frontend(
        self,
        plugin: PluginDefinition[object, object],
        session: Session[object],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        *,
        plots: Plots,
    ) -> InteractiveFrontend:
        """Attach the existing picker UI; the analysis owner settles its session."""
        return make_flux_pick_frontend(
            plugin, session, env, request_finish, request_cancel, plots=plots
        )

    def get_writeback_items(
        self, req: WritebackRequest[RunRecord[C, MuxFluxResult], MuxFluxPickResult]
    ) -> Sequence[WritebackItem]:
        """Propose only this analysis's Q calibration using the Run capture.

        Missing capture yields no proposals. Identity mismatches raise instead
        of guessing a target. Global proposes one complete Q calibration record,
        creating it in an empty mapping without replacing other Q records.
        Local proposals target the selected Q's three scalar fields.
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
            parameters = req.ctx.parameters
            if parameters is None:
                return []
            return [
                ParameterWriteback(
                    target_name=f"{parameters.resolve_path('$mux')}.coil_flux.{picked.qubit}",
                    description=f"{picked.qubit} coil calibration (native DC units)",
                    proposed_value={
                        "half": picked.flx_half,
                        "int": picked.flx_int,
                        "period": picked.flx_period,
                    },
                    element=identity,
                )
            ]
        return [
            ParameterWriteback(
                target_name=f"{picked.qubit}.flux_{name}",
                description=description,
                proposed_value=value,
                element=identity,
            )
            for name, description, value in (
                ("half", "Half-flux DC device value", picked.flx_half),
                ("int", "Integer-flux DC device value", picked.flx_int),
                ("period", "DC device flux period", picked.flx_period),
            )
        ]
