from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)

from .form import T2RamseyInput, make_form

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = RunRecord[T2RamseyCfg, T2RamseyResult]


class T2RamseySummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    t2r: float | None = Field(
        description="Owner t2r in us; null retains an unavailable estimate."
    )

    t2r_err: float | None = Field(
        description="Owner t2r_err in us; null retains an unavailable estimate."
    )

    detune: float | None = Field(
        description="Owner detune in MHz; null retains an unavailable estimate."
    )

    fit_fringe: bool = Field(description="Owner fit_fringe.")


@dataclass
class T2RamseyAnalyzeResult:
    t2r: float
    t2r_err: float
    # ``detune`` is the fitted fringe frequency (MHz); meaningful only when the
    # fringe fit ran (``fit_fringe`` True). Decay-only fits report 0.0.
    detune: float
    fit_fringe: bool
    fit_quality: Mapping[str, FitQuality] | None = None


def _T2Ramsey_analyze(
    req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeOptions], *, plots: Plots
) -> T2RamseyAnalyzeResult:
    params = req.analyze_params
    analysis = T2RamseyExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    return T2RamseyAnalyzeResult(
        t2r=analysis.t2r,
        t2r_err=analysis.t2r_err,
        fit_quality=analysis.fit_quality,
        detune=analysis.detune,
        fit_fringe=params.fit_fringe,
    )


def _T2Ramsey_get_writeback_items(
    req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult],
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
        source_freq = snapshot.pi2_pulse.freq
        if not isinstance(source_freq, float):
            raise ValueError(
                "qubit.freq writeback requires a scalar source pi/2 frequency"
            )
        q_f = source_freq + req.run_result.result.true_activate_detune - result.detune
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
                coordinate_device="$qubit.flux_source",
                proposed_value={
                    "t2r_us": result.t2r,
                    "t2r_err_us": result.t2r_err,
                },
            )
        )
    return items


def _T2Ramsey_make_form(ctx: FormContext) -> Form[T2RamseyInput]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(
    inp: T2RamseyInput, *, context: RunContext
) -> RunRecord[T2RamseyCfg, T2RamseyResult]:
    """Convert fringes/step to MHz, acquire and record the actual detune cfg.

    Zero ratio disables detune. Nonzero ratio on a zero-step delay sweep fails
    before acquisition. The measured, hardware-rounded detune is logged in MHz.
    """
    if inp.detune_ratio == 0.0:
        detune = 0.0
    elif inp.length.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    else:
        detune = inp.detune_ratio / inp.length.step
    cfg = T2RamseyCfg(
        reps=inp.reps,
        rounds=inp.rounds,
        acquire_retry=inp.acquire_retry,
        earlystop_snr=inp.earlystop_snr,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        readout=inp.readout,
        length=inp.length,
        detune=detune,
    )
    result = T2RamseyExp().run(cfg, context=context)
    logger.info("T2Ramsey true detune: %s MHz", result.true_activate_detune)
    return RunRecord(cfg=cfg, result=result)


DEFINITION = Adapter(
    spec=T2RamseyExp.AXES_SPEC,
    make_form=_T2Ramsey_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T2Ramsey_analyze,
        options=lambda source, env: T2RamseyAnalyzeOptions(),
        writeback=_T2Ramsey_get_writeback_items,
    ),
)
