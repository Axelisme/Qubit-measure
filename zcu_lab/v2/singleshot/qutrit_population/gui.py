"""Measure GUI wiring for three preparation-label references."""

from collections.abc import Sequence
from pathlib import Path

from zcu_tools.analysis.singleshot import ReferenceIQAnalysis
from zcu_tools.cfg_model import Centroid3Classifier
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.writeback import calibrated_module_writeback
from zcu_tools.plotting.plots import Plots

from .core import QutritPopulationCfg, QutritPopulationExp, QutritPopulationResult
from .form import make_form


def _run(
    cfg: QutritPopulationCfg, *, context: RunContext
) -> RunRecord[QutritPopulationCfg, QutritPopulationResult]:
    return RunRecord(cfg=cfg, result=QutritPopulationExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[QutritPopulationCfg, QutritPopulationResult], None],
    *,
    plots: Plots,
) -> ReferenceIQAnalysis:
    return QutritPopulationExp().analyze(req.run_result, None, plots=plots)


def _writeback(
    req: WritebackRequest[
        RunRecord[QutritPopulationCfg, QutritPopulationResult], ReferenceIQAnalysis
    ],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    instruction = (
        "Save and bind the readout library module, then Run again before Analyze/Apply."
    )
    if req.ctx.parameters is None or req.run_result.cfg is None:
        raise ValueError(
            f"Qutrit calibration requires a captured readout target. {instruction}"
        )
    classifier = Centroid3Classifier(
        g_center=complex(result.centroids[0]),
        e_center=complex(result.centroids[1]),
        f_center=complex(result.centroids[2]),
        confusion_matrix=result.confusion_matrix.tolist(),
    )
    try:
        item = calibrated_module_writeback(
            req.ctx.parameters,
            req.run_result.cfg.readout,
            target="$resonator.module.readout",
            description="Qutrit g/e/f centers and preparation-label training confusion",
            field_updates={"classifier": classifier},
        )
    except ValueError as error:
        raise ValueError(
            f"Qutrit calibration cannot update the captured readout: {error}. {instruction}"
        ) from error
    return [item]


DEFINITION = Adapter(
    spec=QutritPopulationExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_writeback),
)
