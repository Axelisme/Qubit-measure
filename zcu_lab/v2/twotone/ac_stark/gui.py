"""Measure GUI adapters for AC Stark spectroscopy and Ramsey maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_chi_kappa

from .core import (
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkRamseyAnalyzeOptions,
    AcStarkRamseyCfg,
    AcStarkRamseyExp,
    AcStarkRamseyResult,
    AcStarkResult,
)
from .form import make_form, make_ramsey_form

AcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]
AcStarkRamseyRunResult: TypeAlias = RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult]


@dataclass
class AcStarkAnalyzeParams:
    """Polynomial degree in gain squared and optional exclusive gain cutoff."""

    deg: Annotated[Literal[1, 2], ParamMeta(label="Polynomial degree")] = 1
    cutoff: Annotated[float | None, ParamMeta(label="Gain cutoff (a.u.)")] = None


@dataclass
class AcStarkAnalyzeResult(AnalyzeResultBase):
    """Mean photon number coefficient: n = ac_coeff * Stark gain squared."""

    ac_coeff: float


@dataclass
class AcStarkRamseyAnalyzeParams:
    """FFT center in MHz (subtracted from the axis) and exclusive gain cutoff."""

    detune: Annotated[float, ParamMeta(label="FFT detune (MHz)")] = 0.0
    cutoff: Annotated[float | None, ParamMeta(label="Gain cutoff (a.u.)")] = None


def _AcStark_make_form(ctx: FormContext) -> Form[AcStarkCfg]:
    """Build the spectroscopy form from captured Open/Reset sources."""
    return make_form(ctx)


def _AcStark_analyze(
    req: AnalyzeRequest[AcStarkRunResult, AcStarkAnalyzeParams],
    *,
    plots: Plots,
) -> AcStarkAnalyzeResult:
    """Fit the Stark slope with chi/kappa from this Run/Load capture in MHz.

    Missing calibration fails explicitly. The core publishes the fit figure.
    """
    chi, kappa = read_chi_kappa(req.parameters)
    params = req.analyze_params
    analysis = AcStarkExp().analyze(
        req.run_result,
        AcStarkAnalyzeOptions(
            chi=chi, kappa=kappa, deg=params.deg, cutoff=params.cutoff
        ),
        plots=plots,
    )
    return AcStarkAnalyzeResult(ac_coeff=analysis.ac_coeff)


def _AcStark_get_writeback_items(
    req: WritebackRequest[AcStarkRunResult, AcStarkAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose the photon calibration; Apply remains an explicit user action."""
    return [
        ParameterWriteback(
            target_name="$qubit.ext.ac_stark_coeff",
            description="AC-Stark coefficient (photon number per gain²)",
            proposed_value=req.analyze_result.ac_coeff,
        )
    ]


DEFINITION = Adapter(
    spec=AcStarkExp.AXES_SPEC,
    make_form=_AcStark_make_form,
    run=AcStarkExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AcStark_analyze,
        options=lambda source, env: AcStarkAnalyzeParams(),
        writeback=_AcStark_get_writeback_items,
    ),
)


def _AcStarkRamsey_make_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
    """Build the Ramsey form from captured Open/Reset sources."""
    return make_ramsey_form(ctx)


def _AcStarkRamsey_get_analyze_params(
    result: AcStarkRamseyRunResult, ctx: SessionEnv
) -> AcStarkRamseyAnalyzeParams:
    """Seed the FFT center from the acquired cfg, or zero for cfg-free data."""
    del ctx
    return AcStarkRamseyAnalyzeParams(
        detune=result.cfg.detune if result.cfg is not None else 0.0
    )


def _AcStarkRamsey_analyze(
    req: AnalyzeRequest[AcStarkRamseyRunResult, AcStarkRamseyAnalyzeParams],
    *,
    plots: Plots,
) -> None:
    """Publish contrast/FFT maps, without fitting or proposing calibration."""
    params = req.analyze_params
    AcStarkRamseyExp().analyze(
        req.run_result,
        AcStarkRamseyAnalyzeOptions(detune=params.detune, cutoff=params.cutoff),
        plots=plots,
    )
    return


RAMSEY_DEFINITION = Adapter(
    spec=AcStarkRamseyExp.AXES_SPEC,
    make_form=_AcStarkRamsey_make_form,
    run=AcStarkRamseyExp().run,
    guide=Path(__file__).with_name("guide.ramsey.md"),
    analysis=Analysis(
        fn=_AcStarkRamsey_analyze, options=_AcStarkRamsey_get_analyze_params
    ),
)
