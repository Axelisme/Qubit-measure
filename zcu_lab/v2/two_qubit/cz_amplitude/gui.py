"""Fixed-capture CZ amplitude calibration and operator-selected pulse writeback."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.parameters import DocumentSetCapture
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import module_writeback

from .core import (
    CZAmplitudeAnalysis,
    CZAmplitudeCfg,
    CZAmplitudeExp,
    CZAmplitudeResult,
    CZReturnSummary,
)
from .form import make_form

CZAmplitudeRun: TypeAlias = RunRecord[CZAmplitudeCfg, CZAmplitudeResult]


def _analyze(
    req: AnalyzeRequest[CZAmplitudeRun, None], *, plots: Plots
) -> CZAmplitudeAnalysis:
    """Use saved cfg/counts for joint return, fixed-phase fit and named plots."""
    return CZAmplitudeExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[CZAmplitudeRun, CZAmplitudeAnalysis],
) -> Sequence[WritebackItem]:
    return amplitude_writeback(
        req.run_result.cfg, req.analyze_result.fit, req.ctx.parameters
    )


def amplitude_writeback(
    cfg: CZAmplitudeCfg | None,
    fit: CZReturnSummary,
    parameters: DocumentSetCapture | None,
) -> Sequence[WritebackItem]:
    """Propose two ordinary coupler pulses and two separate resolved slot keys.

    Only source pulse gains change, multiplied by the same valid fitted scale.
    Preserve generator, waveform, frequency, timing, phases and signed ratio.
    Missing Run/Load capture raises ValueError. With a capture, missing cfg
    or an invalid fit returns no items. Read no current library calibration, perform
    no hardware or persistent writes, and leave source cfg untouched.
    """
    if parameters is None:
        raise ValueError("CZ amplitude writeback requires captured parameters")
    if cfg is None or fit.scale is None:
        return []
    coupler = parameters.resolve_path("$coupler")
    items: list[WritebackItem] = []
    for role, pulse in (("qubit1", cfg.drive.qubit1), ("qubit2", cfg.drive.qubit2)):
        # Pulse library keys: $coupler.pulses.cz_qubit1 and $coupler.pulses.cz_qubit2.
        target = f"{coupler}.pulses.cz_{role}"
        items.append(
            module_writeback(
                pulse,
                target=target,
                description=f"CZ {role.capitalize()} first-cycle amplitude",
                field_updates=(("gain", float(pulse.gain) * fit.scale),),
            )
        )
        items.append(
            ParameterWriteback(
                # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
                target_name=f"{coupler}.module.cz_{role}",
                description=f"CZ {role.capitalize()} pulse slot",
                proposed_value=parameters.library_key(target),
            )
        )
    return items


def run(
    cfg: CZAmplitudeCfg, *, context: RunContext
) -> RunRecord[CZAmplitudeCfg, CZAmplitudeResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=CZAmplitudeExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=CZAmplitudeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
