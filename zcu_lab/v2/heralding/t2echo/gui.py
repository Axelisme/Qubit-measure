"""GUI wiring and captured-source proposals for heralded Hahn Echo."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias

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
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalysis,
    T2EchoAnalyzeOptions,
)
from zcu_lab.v2.twotone.time_domain.t2echo.gui import coherence_review_note

from .core import HeraldingT2EchoCfg, HeraldingT2EchoExp, HeraldingT2EchoResult
from .form import HeraldingT2EchoInput, make_form

logger = logging.getLogger(__name__)


HeraldingT2EchoRunResult: TypeAlias = RunRecord[
    HeraldingT2EchoCfg, HeraldingT2EchoResult
]


def analyze(
    req: AnalyzeRequest[HeraldingT2EchoRunResult, T2EchoAnalyzeOptions], *, plots: Plots
) -> T2EchoAnalysis:
    """Fit finite selected time/IQ with the ordinary Echo analysis owner."""
    return HeraldingT2EchoExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def writeback(
    req: WritebackRequest[HeraldingT2EchoRunResult, T2EchoAnalysis],
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
    metadata = req.run_result.metadata
    devices = (
        None
        if metadata is None or metadata.device_snapshot is None
        else deepcopy(dict(metadata.device_snapshot))
    )
    if devices is None:
        logging.warning("no device snapshot")
    result = req.analyze_result
    quality = None if result.fit_quality is None else result.fit_quality.get("fit")
    note = coherence_review_note(result.t2e, result.t2e_err, quality)
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.t2e",
            description="Heralded T2 Echo time (us)",
            proposed_value=result.t2e,
            selected=note is None,
            note=note,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None and metadata is not None and devices is not None:
        frame = record_flux_frame(captured_qubit_flux(parameters), devices)
        items.append(
            RecordWriteback(
                target_name=parameters.resolve_path("$qubit"),
                description="Heralded T2 Echo record at the captured bias",
                record={
                    "devices": {name: deepcopy(info) for name, info in devices.items()},
                    "flux_frame": frame,
                    "run": {
                        "run_id": metadata.run_id,
                        "experiment": metadata.experiment,
                    },
                    "t2e_us": result.t2e,
                    "t2e_err_us": result.t2e_err,
                },
                selected=note is None,
                note=note,
            )
        )
    return items


def run(
    inp: HeraldingT2EchoInput, *, context: RunContext
) -> RunRecord[HeraldingT2EchoCfg, HeraldingT2EchoResult]:
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
    cfg = HeraldingT2EchoCfg(
        reps=inp.reps,
        rounds=inp.rounds,
        acquire_retry=inp.acquire_retry,
        earlystop_snr=inp.earlystop_snr,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        pi_pulse=inp.pi_pulse,
        readout=inp.readout,
        length=inp.length,
        detune=detune,
    )
    result = HeraldingT2EchoExp().run(cfg, context=context)
    logger.info("T2Echo true detune: %s MHz", result.true_activate_detune)
    return RunRecord(cfg=cfg, result=result)


DEFINITION = Adapter(
    spec=HeraldingT2EchoExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: T2EchoAnalyzeOptions(),
        writeback=writeback,
    ),
)
