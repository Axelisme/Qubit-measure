"""GUI wiring for ground-heralded amplitude Rabi."""

from dataclasses import replace
from pathlib import Path

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

from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalysis,
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiResult,
)
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import DEFINITION as AMP_RABI

from .core import HeraldingAmpRabiCfg, HeraldingAmpRabiExp, HeraldingAmpRabiResult
from .form import make_form


def run(
    cfg: HeraldingAmpRabiCfg, *, context: RunContext
) -> RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult]:
    """Acquire selected IQ and retain the acquisition cfg, including classifier."""
    return RunRecord(cfg=cfg, result=HeraldingAmpRabiExp().run(cfg, context=context))


def analyze(
    req: AnalyzeRequest[
        RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult], AmpRabiAnalyzeOptions
    ],
    *,
    plots: Plots,
) -> AmpRabiAnalysis:
    """Fit selected finite points with the ordinary amplitude-Rabi analysis owner."""
    return HeraldingAmpRabiExp().analyze(
        req.run_result, req.analyze_params, plots=plots
    )


def writeback(
    req: WritebackRequest[
        RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult], AmpRabiAnalysis
    ],
) -> list[WritebackItem]:
    """Reuse ordinary Rabi's five proposals, with Heralded descriptions.

    Propose pi_amp/pi2_amp pulses, x180/x90 slots and ext.pi_product at the
    same captured destinations. Without saved cfg return no proposals. Missing
    captured parameters or invalid scalar pulse length fails in the ordinary
    owner. Conversion retains the source pulse; no calibration is applied.
    """
    source = req.run_result
    if source.cfg is None:
        return []
    ordinary = RunRecord(
        cfg=AmpRabiCfg.model_validate(source.cfg.model_dump()),
        result=AmpRabiResult(
            source.result.amps, split_sums(source.result.sums)[1][..., 0]
        ),
        metadata=source.metadata,
    )
    items = AMP_RABI.get_writeback_items(
        WritebackRequest(ordinary, req.analyze_result, req.ctx)
    )
    return [replace(item, description=f"Heralded {item.description}") for item in items]


DEFINITION = Adapter(
    spec=HeraldingAmpRabiExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: AmpRabiAnalyzeOptions(),
        writeback=writeback,
    ),
)
