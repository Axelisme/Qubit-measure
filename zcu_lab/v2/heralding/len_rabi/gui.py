"""GUI wiring and ordinary length-Rabi writeback targets for heralded data."""

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
from zcu_tools.program.v2 import split_sums

from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalysis,
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiResult,
)
from zcu_lab.v2.twotone.rabi.len_rabi.gui import DEFINITION as LEN_RABI

from .core import HeraldingLenRabiCfg, HeraldingLenRabiExp, HeraldingLenRabiResult
from .form import make_form

LenRabiRunResult: TypeAlias = RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]


def analyze(
    req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeOptions], *, plots: Plots
) -> LenRabiAnalysis:
    """Fit finite selected IQ through the existing length-Rabi owner."""
    return HeraldingLenRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def writeback(
    req: WritebackRequest[LenRabiRunResult, LenRabiAnalysis],
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
            else LenRabiCfg.model_validate(source.cfg.model_dump())
        ),
        result=LenRabiResult(
            source.result.lengths, split_sums(source.result.sums)[1][..., 0]
        ),
        metadata=source.metadata,
    )
    items = LEN_RABI.get_writeback_items(
        WritebackRequest(ordinary, req.analyze_result, req.ctx)
    )
    return [replace(item, description=f"Heralded {item.description}") for item in items]


def _run(
    cfg: HeraldingLenRabiCfg, *, context: RunContext
) -> RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]:
    """Keep the unswept cfg and its readout classifier with selected results."""
    return RunRecord(cfg=cfg, result=HeraldingLenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=HeraldingLenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: LenRabiAnalyzeOptions(),
        writeback=writeback,
    ),
)
