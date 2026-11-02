"""Heralded length Rabi proposes pi_len/pi2_len pulses and x180/x90 links."""

from __future__ import annotations

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

from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalysis,
    LenRabiCfg,
    LenRabiResult,
)
from zcu_lab.v2.twotone.rabi.len_rabi.gui import DEFINITION as LEN_RABI

from .core import (
    HeraldingLenRabiAnalysis,
    HeraldingLenRabiAnalyzeOptions,
    HeraldingLenRabiCfg,
    HeraldingLenRabiExp,
    HeraldingLenRabiResult,
)
from .form import make_form

LenRabiRunResult: TypeAlias = RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]


def analyze(
    req: AnalyzeRequest[LenRabiRunResult, HeraldingLenRabiAnalyzeOptions],
    *,
    plots: Plots,
) -> HeraldingLenRabiAnalysis:
    """Fit self-calibrated C and publish fit parameters and population diagnostics."""
    return HeraldingLenRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def writeback(
    req: WritebackRequest[LenRabiRunResult, HeraldingLenRabiAnalysis],
) -> Sequence[WritebackItem]:
    """Reuse ordinary writeback rules and captured Run facts, labeled Heralded.

    Convert only cfg/result types and retain metadata, including device snapshot
    and run identity. Missing cfg/capture follows the ordinary adapter policy.
    Pulse proposals target qubit.pulses.pi_len/pi2_len, and module.x180/x90
    point to those keys. No calibration is applied here.
    """
    source = req.run_result
    ordinary = RunRecord(
        cfg=(
            None
            if source.cfg is None
            else LenRabiCfg.model_validate(source.cfg.model_dump())
        ),
        # Ordinary-record unconditional IQ mapping; writeback does not read signals.
        result=LenRabiResult(
            source.result.lengths,
            (source.result.sums[:, 0, 1] + 1j * source.result.sums[:, 0, 2])
            / source.result.sums[:, 0, 0],
        ),
        metadata=source.metadata,
    )
    analysis = req.analyze_result
    ordinary_analysis = LenRabiAnalysis(
        pi_len=analysis.pi_len,
        pi_len_err=analysis.pi_len_err,
        pi2_len=analysis.pi2_len,
        pi2_len_err=analysis.pi2_len_err,
        rabi_f=analysis.rabi_f,
        rabi_f_err=analysis.rabi_f_err,
        fit_quality=analysis.fit_quality,
    )
    items = LEN_RABI.get_writeback_items(
        WritebackRequest(ordinary, ordinary_analysis, req.ctx)
    )
    return [replace(item, description=f"Heralded {item.description}") for item in items]


def _run(
    cfg: HeraldingLenRabiCfg, *, context: RunContext
) -> RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]:
    """Retain acquisition cfg with unclassified herald/signal second moments."""
    return RunRecord(cfg=cfg, result=HeraldingLenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=HeraldingLenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: HeraldingLenRabiAnalyzeOptions(),
        writeback=writeback,
    ),
)
