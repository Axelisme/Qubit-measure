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
    T2EchoCfg,
    T2EchoResult,
)
from zcu_lab.v2.twotone.time_domain.t2echo.gui import DEFINITION as T2_ECHO

from .core import (
    HeraldingT2EchoAnalysis,
    HeraldingT2EchoAnalyzeOptions,
    HeraldingT2EchoCfg,
    HeraldingT2EchoExp,
    HeraldingT2EchoResult,
)
from .form import HeraldingT2EchoInput, make_form

logger = logging.getLogger(__name__)


HeraldingT2EchoRunResult: TypeAlias = RunRecord[
    HeraldingT2EchoCfg, HeraldingT2EchoResult
]


def analyze(
    req: AnalyzeRequest[HeraldingT2EchoRunResult, HeraldingT2EchoAnalyzeOptions],
    *,
    plots: Plots,
) -> HeraldingT2EchoAnalysis:
    """Fit self-calibrated C and publish fit parameters and population diagnostics."""
    return HeraldingT2EchoExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def writeback(
    req: WritebackRequest[HeraldingT2EchoRunResult, HeraldingT2EchoAnalysis],
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
        # Ordinary-record unconditional IQ mapping; writeback does not read signals.
        result=T2EchoResult(
            source.result.times,
            split_sums(source.result.sums[:, 0, :])[1],
            source.result.true_activate_detune,
        ),
        metadata=source.metadata,
    )
    analysis = req.analyze_result
    ordinary_analysis = T2EchoAnalysis(
        t2e=analysis.t2e,
        t2e_err=analysis.t2e_err,
        detune=analysis.detune,
        detune_err=analysis.detune_err,
        fit_quality=analysis.fit_quality,
    )
    items = T2_ECHO.get_writeback_items(
        WritebackRequest(ordinary, ordinary_analysis, req.ctx)
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
        options=lambda source, env: HeraldingT2EchoAnalyzeOptions(),
        writeback=writeback,
    ),
)
