from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)

from .form import make_form

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = RunRecord[T2RamseyCfg, T2RamseyResult]


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


@dataclass
class T2RamseyAnalyzeResult(AnalyzeResultBase):
    t2r: float
    t2r_err: float
    # ``detune`` is the fitted fringe frequency (MHz); meaningful only when the
    # fringe fit ran (``fit_fringe`` True). Decay-only fits report 0.0.
    detune: float
    fit_fringe: bool
    fit_quality: dict[str, dict[str, object]] | None = None


class T2RamseyAdapter(
    BaseAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T2RamseyCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = T2RamseyExp.persistence_tag()
    sample_device_alias: ClassVar[str | None] = "$flux_dev"
    exp_cls = T2RamseyExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: T2RamseyCfg,
        *,
        context: RunContext,
    ) -> T2RamseyRunResult:
        del req
        result = self.exp_cls().run(cfg, context=context)
        logger.info("T2Ramsey true detune: %s MHz", result.true_activate_detune)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeParams],
        *,
        plots: Plots,
    ) -> T2RamseyAnalyzeResult:
        params = req.analyze_params
        analysis = T2RamseyExp().analyze(
            req.run_result,
            T2RamseyAnalyzeOptions(
                fit_fringe=params.fit_fringe, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        return T2RamseyAnalyzeResult(
            t2r=analysis.t2r,
            t2r_err=analysis.t2r_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
            detune=analysis.detune,
            fit_fringe=params.fit_fringe,
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose captured qubit.t2r, plus qubit.freq for a sourced fringe fit.

        req supplies the committed fit, source record and captured context.
        Frequency is source pi/2 frequency + realized detune - fitted detune;
        it is omitted for decay fits, cfg=None or unknown realized detune.
        Non-scalar source frequency raises ValueError. WritebackService resolves
        destinations on req.ctx.parameters and rejects missing captures/unbound
        roles. With a capture, also propose t2r_us/t2r_err_us for the Run's
        resolved $qubit; an unbound alias raises RoleResolutionError. Without
        a capture, retain parameter proposals only so loaded data can still be
        analyzed. The service adds bias coordinates or omits the sample with a
        public reason. This Run leaves devices fixed; no live calibration lookup,
        device access or write occurs here.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t2r",
                description="T2 Ramsey time (us)",
                proposed_value=result.t2r,
            ),
        ]

        # Frequency correction uses only the source run, never current calibration.
        snapshot = req.run_result.cfg
        if (
            result.fit_fringe
            and snapshot is not None
            and req.run_result.result.true_activate_detune is not None
        ):
            source_freq = snapshot.modules.pi2_pulse.freq
            if not isinstance(source_freq, float):
                raise ValueError(
                    "qubit.freq writeback requires a scalar source pi/2 frequency"
                )
            q_f = (
                source_freq + req.run_result.result.true_activate_detune - result.detune
            )
            items.append(
                ParameterWriteback(
                    target_name="$qubit.freq",
                    description="Qubit frequency (MHz)",
                    proposed_value=q_f,
                )
            )

        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="T2 Ramsey sample at the captured bias",
                    proposed_value={
                        "t2r_us": result.t2r,
                        "t2r_err_us": result.t2r_err,
                    },
                )
            )
        return items
