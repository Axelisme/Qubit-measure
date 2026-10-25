"""Save mux members and apply overnight scalar acceptance to native tone fits."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Completed, Failed
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.mux.t1.core import MuxT1Cfg, MuxT1Exp, MuxT1Result
from zcu_lab.v2.mux.t2echo.core import MuxT2EchoCfg, MuxT2EchoExp
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_coherence import MuxCoherenceResult
from zcu_lab.workflows.common.analysis import lifetime_rejection
from zcu_lab.workflows.common.saving import save_native

if TYPE_CHECKING:
    from .workflow import OvernightMuxPlan, OvernightMuxPoint, OvernightMuxTunables


def _summary(
    lifetime: float,
    error: float,
    quality: FitQuality | None,
    tun: OvernightMuxTunables,
    name: str,
    times: NDArray[np.float64],
) -> tuple[float | None, float | None, str | None]:
    rejection = lifetime_rejection(quality, times, lifetime, tun.min_lifetime_points)
    if rejection is not None:
        return None, None, f"{name} fit rejected: {rejection}"
    residual = quality.normalized_residual_rms if quality is not None else None
    if not (
        isfinite(lifetime)
        and lifetime > 0
        and isfinite(error)
        and error >= 0
        and error / lifetime < tun.max_rel_err
        and residual is not None
        and isfinite(residual)
        and residual < tun.max_fit_residual
    ):
        return None, None, f"{name} fit rejected"
    return lifetime, error, None


def record_t1(
    iteration: Scope,
    record: Completed[RunRecord[MuxT1Cfg, MuxT1Result]] | Failed,
    point: OvernightMuxPoint,
    plan: OvernightMuxPlan,
    tun: OvernightMuxTunables,
    plots: Plots,
) -> None:
    """Save Completed T1 raw then summarize each native tone fit in point.

    Failed/partial members fill all participants' reasons without fitting.
    plan supplies historical snapshot and original participant indices; plots
    supplies the host for temporary native fit_i figures, released after analysis.
    Rejected fits keep the native filename. Writer, structural analysis and plot
    errors propagate; no acquisition occurs here.
    """
    if isinstance(record, Failed):
        for qubit in point.qubits:
            qubit.t1_reason = record.reason
        return
    save_native(
        record,
        iteration.dir / "t1.h5",
        spec=MuxT1Exp.AXES_SPEC,
        snapshot=plan.snapshot,
        identity=iteration.identity,
    )
    point.t1_file = "t1.h5"
    if record.partial_result:
        for qubit in point.qubits:
            qubit.t1_reason = "stopped"
        return
    # Native analyzers create fit_i figures afresh; keep them out of the trend
    # collection so Echo and later rounds cannot collide with the same names.
    fit_plots = plots.fork()
    try:
        fits = MuxT1Exp().analyze(
            record.value, MuxT1Exp.Options(skip=0, dual_exp=False), plots=fit_plots
        )
    finally:
        fit_plots.finish(present=False)
        fit_plots.release()
    qubits = dict(zip(plan.participants, point.qubits, strict=True))
    for row, fit in enumerate(fits):
        qubit = qubits[fit.index]
        if isinstance(fit, MuxFitFailure):
            qubit.t1_reason = fit.reason
        else:
            quality = (
                fit.fit.fit_quality["fit"] if fit.fit.fit_quality is not None else None
            )
            qubit.t1_us, qubit.t1_error_us, qubit.t1_reason = _summary(
                fit.fit.t1,
                fit.fit.t1_err,
                quality,
                tun,
                "T1",
                record.value.result.times[
                    np.isfinite(record.value.result.signals[row])
                ],
            )


def record_echo(
    iteration: Scope,
    record: Completed[RunRecord[MuxT2EchoCfg, MuxCoherenceResult]] | Failed,
    point: OvernightMuxPoint,
    plan: OvernightMuxPlan,
    tun: OvernightMuxTunables,
    plots: Plots,
) -> None:
    """Save Completed Echo raw then summarize native zero-detune per-tone fits.

    Arguments and failure/partial semantics match record_t1. Native analysis
    omits the first delay sample and fits decay, not fringes. Independent tone
    failures retain their own reasons and do not suppress another tone's fit.
    """
    if isinstance(record, Failed):
        for qubit in point.qubits:
            qubit.t2_reason = record.reason
        return
    save_native(
        record,
        iteration.dir / "t2echo.h5",
        spec=MuxT2EchoExp.AXES_SPEC,
        snapshot=plan.snapshot,
        identity=iteration.identity,
    )
    point.t2_file = "t2echo.h5"
    if record.partial_result:
        for qubit in point.qubits:
            qubit.t2_reason = "stopped"
        return
    fit_plots = plots.fork()
    try:
        fits = MuxT2EchoExp().analyze(
            record.value, MuxT2EchoExp.Options(fit_method="decay"), plots=fit_plots
        )
    finally:
        fit_plots.finish(present=False)
        fit_plots.release()
    qubits = dict(zip(plan.participants, point.qubits, strict=True))
    for row, fit in enumerate(fits):
        qubit = qubits[fit.index]
        if isinstance(fit, MuxFitFailure):
            qubit.t2_reason = fit.reason
        else:
            qubit.t2_us, qubit.t2_error_us, qubit.t2_reason = _summary(
                fit.t2,
                fit.t2_err,
                fit.quality,
                tun,
                "T2 Echo",
                record.value.result.times[1:][
                    np.isfinite(record.value.result.signals[row, 1:])
                ],
            )
