from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadataContent
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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.flux import record_flux_frame
from zcu_lab.v2.support.measure.record_frame import captured_qubit_flux
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalysis,
    T2EchoAnalyzeOptions,
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)

from .form import T2EchoInput, make_form

logger = logging.getLogger(__name__)


T2EchoRunResult: TypeAlias = RunRecord[T2EchoCfg, T2EchoResult]


class T2EchoSummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    fit_quality: Mapping[str, FitQuality] | None = Field(
        description='Owner fit diagnostics under "fit"; null when unavailable.'
    )

    t2e: float | None = Field(
        description="Owner t2e in us; null retains an unavailable estimate."
    )

    t2e_err: float | None = Field(
        description="Owner t2e_err in us; null retains an unavailable estimate."
    )


def _T2Echo_analyze(
    req: AnalyzeRequest[T2EchoRunResult, T2EchoAnalyzeOptions], *, plots: Plots
) -> T2EchoAnalysis:
    return T2EchoExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _T2Echo_get_writeback_items(
    req: WritebackRequest[T2EchoRunResult, T2EchoAnalysis],
) -> Sequence[WritebackItem]:
    """Propose fitted T2E in us and a record for the Run's resolved $qubit.

    req supplies the result and its captured context; source.cfg may be None.
    WritebackService resolves the role and destination on req.ctx.parameters
    and rejects a missing capture or unbound role. Without a capture,
    retain parameter proposals only so loaded data can still be analyzed.
    With a capture, unbound $qubit raises RoleResolutionError. The adapter
    copies Run device facts; missing snapshot logs "no device snapshot" and
    omits only the record. This
    Run leaves devices fixed; no live lookup, device access, frequency/module
    update or write occurs here.
    """
    metadata = RunMetadataContent(req.run_result.metadata)
    devices = (
        None
        if metadata.device_snapshot is None
        else deepcopy(dict(metadata.device_snapshot))
    )
    if devices is None:
        logging.warning("no device snapshot")
    result = req.analyze_result
    note = (
        None if result.review_note is None else f"Review required: {result.review_note}"
    )
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.t2e",
            description="T2 Echo time (µs)",
            proposed_value=result.t2e,
            selected=note is None,
            note=note,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None and devices is not None:
        frame = record_flux_frame(captured_qubit_flux(parameters), devices)
        items.append(
            RecordWriteback(
                target_name=parameters.resolve_path("$qubit"),
                description="T2 Echo record at the captured bias",
                record={
                    "devices": {name: deepcopy(info) for name, info in devices.items()},
                    "flux_frame": frame,
                    "experiment": metadata.experiment,
                    "t2e_us": result.t2e,
                    "t2e_err_us": result.t2e_err,
                },
                selected=note is None,
                note=note,
            )
        )
    return items


def _T2Echo_make_form(ctx: FormContext) -> Form[T2EchoInput]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(
    inp: T2EchoInput, *, context: RunContext
) -> RunRecord[T2EchoCfg, T2EchoResult]:
    """Convert fringes/step to MHz, acquire and record the actual detune cfg.

    Zero ratio disables detune. Nonzero ratio on a zero-step delay sweep fails
    before acquisition. The measured, hardware-rounded detune is logged in MHz.
    """
    if inp.detune_ratio == 0.0:
        detune = 0.0
    elif inp.delay.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    else:
        detune = inp.detune_ratio / inp.delay.step
    cfg = T2EchoCfg(
        reps=inp.reps,
        rounds=inp.rounds,
        earlystop_snr=inp.earlystop_snr,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        pi_pulse=inp.pi_pulse,
        readout=inp.readout,
        delay=inp.delay,
        detune=detune,
    )
    result = T2EchoExp().run(cfg, context=context)
    logger.info("T2Echo true detune: %s MHz", result.true_activate_detune)
    return RunRecord(cfg=cfg, result=result)


def _summary_lines(source: T2EchoRunResult) -> tuple[str, ...]:
    lines: list[str] = []
    if source.cfg is not None:
        lines.append(f"Reps × Rounds: {source.cfg.reps} × {source.cfg.rounds}")
    if source.result.true_activate_detune is not None:
        lines.append(f"Applied detune: {source.result.true_activate_detune:g} MHz")
    return tuple(lines)


DEFINITION = Adapter(
    spec=T2EchoExp.AXES_SPEC,
    make_form=_T2Echo_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T2Echo_analyze,
        summary_lines=_summary_lines,
        options=lambda source, env: T2EchoAnalyzeOptions(),
        writeback=_T2Echo_get_writeback_items,
    ),
)
