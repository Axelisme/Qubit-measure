"""GUI wiring and captured-source writeback for heralded Ramsey."""

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

from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyResult,
)
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import (
    DEFINITION as T2_RAMSEY,
)
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAnalyzeResult

from .core import (
    HeraldingT2RamseyCfg,
    HeraldingT2RamseyExp,
    HeraldingT2RamseyResult,
)
from .form import HeraldingT2RamseyInput, make_form

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = RunRecord[HeraldingT2RamseyCfg, HeraldingT2RamseyResult]


def _T2Ramsey_analyze(
    req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeOptions], *, plots: Plots
) -> T2RamseyAnalyzeResult:
    params = req.analyze_params
    analysis = HeraldingT2RamseyExp().analyze(
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
            else T2RamseyCfg.model_validate(source.cfg.model_dump())
        ),
        result=T2RamseyResult(
            source.result.times,
            split_sums(source.result.sums)[1][..., 0],
            source.result.true_activate_detune,
        ),
        metadata=source.metadata,
    )
    items = T2_RAMSEY.get_writeback_items(
        WritebackRequest(ordinary, req.analyze_result, req.ctx)
    )
    return [replace(item, description=f"Heralded {item.description}") for item in items]


def _run(
    inp: HeraldingT2RamseyInput, *, context: RunContext
) -> RunRecord[HeraldingT2RamseyCfg, HeraldingT2RamseyResult]:
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
    cfg = HeraldingT2RamseyCfg(
        reps=inp.reps,
        rounds=inp.rounds,
        acquire_retry=inp.acquire_retry,
        earlystop_snr=inp.earlystop_snr,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        readout=inp.readout,
        delay=inp.delay,
        detune=detune,
    )
    result = HeraldingT2RamseyExp().run(cfg, context=context)
    logger.info("T2Ramsey true detune: %s MHz", result.true_activate_detune)
    return RunRecord(cfg=cfg, result=result)


DEFINITION = Adapter(
    spec=HeraldingT2RamseyExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T2Ramsey_analyze,
        options=lambda source, env: T2RamseyAnalyzeOptions(),
        writeback=_T2Ramsey_get_writeback_items,
    ),
)
