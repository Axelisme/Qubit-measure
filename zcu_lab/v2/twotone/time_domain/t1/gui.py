from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.flux import record_flux_frame
from zcu_lab.v2.support.measure.record_frame import captured_qubit_flux
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)

from .form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext


T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


class T1Summary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )


def _T1_analyze(
    req: AnalyzeRequest[T1RunResult, T1AnalyzeOptions], *, plots: Plots
) -> T1Analysis:
    params = req.analyze_params
    if params.skip < 0:
        raise ValueError("Skip leading points must be nonnegative")
    return T1Exp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _T1_get_writeback_items(
    req: WritebackRequest[T1RunResult, T1Analysis],
) -> Sequence[WritebackItem]:
    """Propose fitted T1 parameters and a captured-component record.

    req supplies the source result and fixed Run context. Record fields use
    us and target the Run's resolved $qubit, never the live selection. Without
    a parameter capture, retain parameter proposals only so loaded data can
    still be analyzed; Apply rejects the missing capture. Unbound $qubit
    with a capture raises RoleResolutionError. The adapter copies all Run
    devices into its record. Missing metadata or snapshot logs "no device snapshot" and omits only the record.
    No write or device access occurs here; this Run leaves devices fixed.
    """
    metadata = req.run_result.metadata
    devices = (
        None
        if metadata is None or metadata.device_snapshot is None
        else deepcopy(dict(metadata.device_snapshot))
    )
    if devices is None:
        logging.warning("no device snapshot")
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.t1",
            description="T1 relaxation time (us)",
            proposed_value=result.t1,
            stderr=result.t1_err,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None and metadata is not None and devices is not None:
        frame = record_flux_frame(captured_qubit_flux(parameters), devices)
        items.append(
            RecordWriteback(
                target_name=parameters.resolve_path("$qubit"),
                description="T1 record at the captured bias",
                record={
                    "devices": {name: deepcopy(info) for name, info in devices.items()},
                    "flux_frame": {
                        name: dict(calibration) for name, calibration in frame.items()
                    },
                    "run": {
                        "run_id": metadata.run_id,
                        "experiment": metadata.experiment,
                    },
                    "t1_us": result.t1,
                    "t1_err_us": result.t1_err,
                },
            )
        )
    return items


def _T1_make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: T1Cfg, *, context: RunContext) -> RunRecord[T1Cfg, T1Result]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=_T1_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T1_analyze,
        options=lambda source, env: T1AnalyzeOptions(),
        writeback=_T1_get_writeback_items,
    ),
)
