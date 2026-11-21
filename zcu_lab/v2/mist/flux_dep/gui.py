"""Measure adapter for the native MIST flux/gain map."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadataContent
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.ctx_helpers import flux_pick_initial_points

from .core import FluxDepAnalyzeOptions, FluxDepCfg, FluxDepExp, FluxDepResult
from .form import make_form

MistFluxDepRun: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


@dataclass
class MistFluxDepParams:
    """Display-only calibration; None keeps raw device/gain coordinates.

    flux_half and flux_period use the acquisition device's native unit.
    Both flux values enable a bottom flux axis and top device-value ticks.
    ac_coeff maps probe gain squared to photon number.
    auto_range uses the full acquired endpoints.
    """

    flux_half: Annotated[
        float | None, ParamMeta(label="Half-flux bias (device units)")
    ] = None
    flux_period: Annotated[
        float | None, ParamMeta(label="Flux period (device units / Φ₀)")
    ] = None
    ac_coeff: Annotated[
        float | None, ParamMeta(label="Photon number / gain squared")
    ] = None
    auto_range: Annotated[bool, ParamMeta(label="Full acquired range")] = True


@dataclass
class MistFluxDepAnalysis:
    """Figure-only MIST analysis; no calibration or writeback is inferred."""


def _default_params(source: MistFluxDepRun, env: SessionEnv) -> MistFluxDepParams:
    """Prefill editable calibration from the Run capture for its scanned device."""
    cfg = source.cfg
    points = (
        None
        if cfg is None
        else flux_pick_initial_points(
            env.parameters,
            cfg.flux_dev,
            RunMetadataContent(source.metadata).device_snapshot,
        )
    )
    return MistFluxDepParams(
        flux_half=None if points is None else points.half,
        flux_period=None if points is None else points.period,
    )


def _analyze(
    req: AnalyzeRequest[MistFluxDepRun, MistFluxDepParams],
    *,
    plots: Plots,
) -> MistFluxDepAnalysis:
    """Publish fit from the explicit source; missing historical cfg is allowed.

    Display calibration starts from the Run capture and can be edited or cleared
    in the analysis pane. Both flux values enable the calibrated axes.
    """
    params = req.analyze_params
    FluxDepExp().analyze(
        req.run_result,
        FluxDepAnalyzeOptions(
            flux_half=params.flux_half,
            flux_period=params.flux_period,
            ac_coeff=params.ac_coeff,
            auto_range=params.auto_range,
        ),
        plots=plots,
    )
    return MistFluxDepAnalysis()


def run(
    cfg: FluxDepCfg, *, context: RunContext
) -> RunRecord[FluxDepCfg, FluxDepResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=FluxDepExp().run(cfg, context=context))


def _summary_lines(source: MistFluxDepRun) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


DEFINITION = Adapter(
    spec=FluxDepExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=_default_params, summary_lines=_summary_lines
    ),
)
