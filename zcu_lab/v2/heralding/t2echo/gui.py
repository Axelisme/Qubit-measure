"""GUI wiring and captured-source proposals for heralded Hahn Echo."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import split_sums

from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalysis,
    T2EchoAnalyzeOptions,
    T2EchoCfg,
    T2EchoResult,
)
from zcu_lab.v2.twotone.time_domain.t2echo.gui import DEFINITION as T2_ECHO

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
    """Reuse ordinary writeback rules and captured Run facts, labeled Heralded.

    Convert only cfg/result types and retain metadata, including device snapshot
    and run identity. Missing cfg/capture follows the ordinary adapter policy.
    No calibration is applied here.
    """
    source = req.run_result
    ordinary = RunRecord(
        cfg=(
            None
            if source.cfg is None
            else T2EchoCfg.model_validate(source.cfg.model_dump())
        ),
        result=T2EchoResult(
            source.result.times,
            split_sums(source.result.sums)[1][..., 0],
            source.result.true_activate_detune,
        ),
        metadata=source.metadata,
    )
    items = T2_ECHO.get_writeback_items(
        WritebackRequest(ordinary, req.analyze_result, req.ctx)
    )
    return [replace(item, description=f"Heralded {item.description}") for item in items]


def run(
    inp: HeraldingT2EchoInput, *, context: RunContext
) -> RunRecord[HeraldingT2EchoCfg, HeraldingT2EchoResult]:
    """Convert fringes/step to MHz, acquire and record the actual detune cfg.

    Zero ratio disables detune. Nonzero ratio on a zero-step delay sweep fails
    before acquisition. The measured, hardware-rounded detune is logged in MHz.
    """
    if inp.detune_ratio == 0.0:
        detune = 0.0
    elif inp.delay.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "delay sweep (expts < 2, step == 0)"
        )
    else:
        detune = inp.detune_ratio / inp.delay.step
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
        delay=inp.delay,
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
