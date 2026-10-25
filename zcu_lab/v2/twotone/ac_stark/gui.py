"""Measure GUI adapters for AC Stark spectroscopy and Ramsey maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.singleshot_shared import read_chi_kappa

from .core import (
    AcStarkAnalysis,
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)
from .form import make_form, make_ramsey_form
from .ramsey_core import (
    AcStarkRamseyAnalyzeOptions,
    AcStarkRamseyCfg,
    AcStarkRamseyExp,
    AcStarkRamseyResult,
)

AcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]


AcStarkRamseyRunResult: TypeAlias = RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult]


@dataclass
class AcStarkAnalyzeParams:
    """Polynomial degree in gain squared and optional exclusive gain cutoff."""

    deg: Annotated[Literal[1, 2], ParamMeta(label="Polynomial degree")] = 1
    cutoff: Annotated[float | None, ParamMeta(label="Gain cutoff (a.u.)")] = None


def _AcStark_analyze(
    req: AnalyzeRequest[AcStarkRunResult, AcStarkAnalyzeParams], *, plots: Plots
) -> AcStarkAnalysis:
    """Fit the Stark slope with chi/kappa from this Run/Load capture in MHz.

    Missing calibration fails explicitly. The core publishes the fit figure.
    """
    chi, kappa = read_chi_kappa(req.parameters)
    params = req.analyze_params
    return AcStarkExp().analyze(
        req.run_result,
        AcStarkAnalyzeOptions(
            chi=chi, kappa=kappa, deg=params.deg, cutoff=params.cutoff
        ),
        plots=plots,
    )


def _AcStark_get_writeback_items(
    req: WritebackRequest[AcStarkRunResult, AcStarkAnalysis],
) -> Sequence[WritebackItem]:
    """Propose the photon calibration; Apply remains an explicit user action."""
    return [
        ParameterWriteback(
            target_name="$qubit.ext.ac_stark_coeff",
            description="AC-Stark coefficient (photon number per gain²)",
            proposed_value=req.analyze_result.ac_coeff,
        )
    ]


def _AcStarkRamsey_get_analyze_params(
    result: AcStarkRamseyRunResult, ctx: SessionEnv
) -> AcStarkRamseyAnalyzeOptions:
    """Seed the FFT center from the acquired cfg, or zero for cfg-free data."""
    del ctx
    return AcStarkRamseyAnalyzeOptions(
        detune=result.cfg.detune if result.cfg is not None else 0.0
    )


def _AcStarkRamsey_analyze(
    req: AnalyzeRequest[AcStarkRamseyRunResult, AcStarkRamseyAnalyzeOptions],
    *,
    plots: Plots,
) -> None:
    """Publish contrast/FFT maps, without fitting or proposing calibration."""
    AcStarkRamseyExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    return


def _AcStark_make_form(ctx: FormContext) -> Form[AcStarkCfg]:
    """Build the spectroscopy form from captured Open/Reset sources."""
    return make_form(ctx)


def _run_acstark(
    cfg: AcStarkCfg, *, context: RunContext
) -> RunRecord[AcStarkCfg, AcStarkResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AcStarkExp().run(cfg, context=context))


def _run_acstarkramsey(
    cfg: AcStarkRamseyCfg, *, context: RunContext
) -> RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AcStarkRamseyExp().run(cfg, context=context))


def _AcStarkRamsey_make_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
    """Build the Ramsey form from captured Open/Reset sources."""
    return make_ramsey_form(ctx)


DEFINITION = Adapter(
    spec=AcStarkExp.AXES_SPEC,
    make_form=_AcStark_make_form,
    run=_run_acstark,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AcStark_analyze,
        options=lambda source, env: AcStarkAnalyzeParams(),
        writeback=_AcStark_get_writeback_items,
    ),
)


RAMSEY_DEFINITION = Adapter(
    spec=AcStarkRamseyExp.AXES_SPEC,
    make_form=_AcStarkRamsey_make_form,
    run=_run_acstarkramsey,
    guide=Path(__file__).with_name("guide.ramsey.md"),
    analysis=Analysis(
        fn=_AcStarkRamsey_analyze, options=_AcStarkRamsey_get_analyze_params
    ),
)
