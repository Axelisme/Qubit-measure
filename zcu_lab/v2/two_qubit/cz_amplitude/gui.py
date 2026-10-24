"""Fixed-capture CZ amplitude calibration and operator-selected pulse writeback."""

from collections.abc import Sequence
from dataclasses import dataclass
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import module_writeback

from .analysis import CZAmplitudeAnalysis
from .core import CZAmplitudeCfg, CZAmplitudeExp, CZAmplitudeResult
from .form import make_form

CZAmplitudeRun: TypeAlias = RunRecord[CZAmplitudeCfg, CZAmplitudeResult]


@dataclass(frozen=True)
class CZAmplitudeAnalyzeResult:
    """JSON-safe projection of saved-source classification and first-cycle fitting."""

    analysis: CZAmplitudeAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Report finite diagnostics, coverage failures and uncertainty without NaN JSON."""
        data, fit = self.analysis.data, self.analysis.fit
        return {
            "scale": fit.scale,
            "period": fit.period,
            "period_error": fit.error,
            "offset": fit.offset,
            "contrast": fit.contrast,
            "reason": fit.reason,
            "fit_quality": fit.fit_quality.to_summary_dict()
            if fit.fit_quality
            else None,
            "complete_scales": int(data.complete.sum()),
            "acquired_shots": data.acquired_shots.tolist(),
            "analyzed_shots": data.analyzed_shots.tolist(),
            "joint_counts": [
                row.astype(int).tolist() if complete else None
                for row, complete in zip(data.counts, data.complete, strict=True)
            ],
            "joint_probabilities": [
                row.tolist() if complete else None
                for row, complete in zip(data.probabilities, data.complete, strict=True)
            ],
            "interpretation": "Zero-phase single-frequency model; first positive return requires measured 0 through the full period. Thermal nominal B11; same-shot return is a leakage proxy, not physical gate fidelity. Large-amplitude AC Stark shifts can violate the model. No heralding, radius cuts, confusion inversion or automatic statistical threshold.",
        }


def _analyze(
    req: AnalyzeRequest[CZAmplitudeRun, None], *, plots: Plots
) -> CZAmplitudeAnalyzeResult:
    """Use only saved cfg/raw for joint return, fixed-phase fit and named plots."""
    return CZAmplitudeAnalyzeResult(
        CZAmplitudeExp().analyze(req.run_result, None, plots=plots)
    )


def _get_writeback_items(
    req: WritebackRequest[CZAmplitudeRun, CZAmplitudeAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose two ordinary coupler pulses and two separate resolved slot keys.

    Only source pulse gains change, multiplied by the same valid fitted scale.
    Preserve generator, waveform, frequency, timing, phases and signed ratio.
    Missing cfg or an invalid fit returns no items. Nonempty proposals require
    the Run/Load's capture with matching role identities; absent/mismatched
    capture raises ValueError. Read no current library calibration, perform
    no hardware or persistent writes, and leave source cfg untouched.
    """
    cfg, fit = req.run_result.cfg, req.analyze_result.analysis.fit
    if cfg is None or fit.scale is None:
        return []
    parameters = req.ctx.parameters
    if parameters is None:
        raise ValueError("CZ amplitude writeback requires captured parameters")
    for role, identity in (
        ("f1", cfg.roles.f1),
        ("f2", cfg.roles.f2),
        ("coupler", cfg.roles.coupler),
        ("mux", cfg.roles.mux),
    ):
        if parameters.resolve_path(f"${role}") != identity:
            raise ValueError(
                "CZ amplitude writeback capture must match saved source roles"
            )
    items: list[WritebackItem] = []
    for role, pulse in (("f1", cfg.drive.f1), ("f2", cfg.drive.f2)):
        target = f"$coupler.pulses.cz_{role}"
        items.append(
            module_writeback(
                pulse,
                target=target,
                description=f"CZ {role.upper()} first-cycle amplitude",
                field_updates=(("gain", float(pulse.gain) * fit.scale),),
            )
        )
        items.append(
            ParameterWriteback(
                target_name=f"$coupler.module.cz_{role}",
                description=f"CZ {role.upper()} pulse slot",
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
