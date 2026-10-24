"""Measure GUI adapters for AC Stark spectroscopy and Ramsey maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_chi_kappa

from .core import (
    AcStarkAnalysis,
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


class AcStarkAdapter(
    BaseAdapter[AcStarkCfg, AcStarkRunResult, AcStarkAnalysis, AcStarkAnalyzeParams]
):
    """Acquire probe-frequency maps and propose the photon/gain-squared calibration."""

    exp_cls = AcStarkExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[AcStarkCfg]:
        """Build the spectroscopy form from captured Open/Reset sources."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: AcStarkCfg, *, context: RunContext
    ) -> AcStarkRunResult:
        """Acquire mean IQ on the Stark gain/probe-frequency grid."""
        del req
        return RunRecord(cfg=cfg, result=self.exp_cls().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[AcStarkRunResult, AcStarkAnalyzeParams],
        *,
        plots: Plots,
    ) -> AcStarkAnalysis:
        """Fit the Stark slope with chi/kappa from this Run/Load capture in MHz.

        Missing calibration fails explicitly. The core publishes the fit figure.
        """
        chi, kappa = read_chi_kappa(req.parameters)
        params = req.analyze_params
        return self.exp_cls().analyze(
            req.run_result,
            AcStarkAnalyzeOptions(
                chi=chi, kappa=kappa, deg=params.deg, cutoff=params.cutoff
            ),
            plots=plots,
        )

    def get_writeback_items(
        self, req: WritebackRequest[AcStarkRunResult, AcStarkAnalysis]
    ) -> Sequence[WritebackItem]:
        """Propose the photon calibration; Apply remains an explicit user action."""
        return [
            ParameterWriteback(
                target_name="$qubit.ext.ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_coeff,
            )
        ]


class AcStarkRamseyAdapter(
    BaseAdapter[
        AcStarkRamseyCfg,
        AcStarkRamseyRunResult,
        None,
        AcStarkRamseyAnalyzeOptions,
    ]
):
    """Acquire Stark-driven Ramsey maps and display the core's normalized FFT."""

    exp_cls = AcStarkRamseyExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.ramsey.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
        """Build the Ramsey form from captured Open/Reset sources."""
        return make_ramsey_form(ctx)

    def run(
        self, req: RunRequest, cfg: AcStarkRamseyCfg, *, context: RunContext
    ) -> AcStarkRamseyRunResult:
        """Acquire mean IQ on the Stark gain/Ramsey-delay grid."""
        del req
        return RunRecord(cfg=cfg, result=self.exp_cls().run(cfg, context=context))

    def get_analyze_params(
        self, result: AcStarkRamseyRunResult, ctx: SessionEnv
    ) -> AcStarkRamseyAnalyzeOptions:
        """Seed the FFT center from the acquired cfg, or zero for cfg-free data."""
        del ctx
        return AcStarkRamseyAnalyzeOptions(
            detune=result.cfg.detune if result.cfg is not None else 0.0
        )

    def analyze(
        self,
        req: AnalyzeRequest[AcStarkRamseyRunResult, AcStarkRamseyAnalyzeOptions],
        *,
        plots: Plots,
    ) -> None:
        """Publish contrast/FFT maps, without fitting or proposing calibration."""
        self.exp_cls().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )
        return
