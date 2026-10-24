"""GUI spectra, frequency fits and calibration proposals for mux qubits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.analyze_results import fit_quality_models
from zcu_lab.v2.support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions
from zcu_lab.v2.twotone.freq.gui import FreqAnalyzeParams

from .analysis import MuxFreqFit
from .core import MuxFreqCfg, MuxFreqExp
from .form import make_form

MuxFreqRunResult: TypeAlias = RunRecord[MuxFreqCfg, MuxSweepResult]


class MuxTwotoneSpectrumFitSummary(MuxSummaryTone):
    """Native fit estimates in experiment units and qualified owner quality."""

    freq: float | None = Field(
        description="Owner freq in MHz; null retains an unavailable estimate."
    )

    freq_err: float | None = Field(
        description="Owner freq_err in MHz; null retains an unavailable estimate."
    )

    fwhm: float | None = Field(
        description="Owner fwhm in MHz; null retains an unavailable estimate."
    )

    fwhm_err: float | None = Field(
        description="Owner fwhm_err in MHz; null retains an unavailable estimate."
    )

    fit_quality: dict[str, FitQuality] | None


class MuxTwotoneSpectrumSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxTwotoneSpectrumFitSummary | MuxSummaryFailure]


@dataclass(frozen=True)
class MuxFreqAnalyzeResult(AnalyzeResultBase):
    """Frequency fits or failures for each measured original tone position.

    outcomes contains per-tone estimates or failure reasons. names is the full
    qubit name list from the measured settings, or None when unavailable.
    """

    outcomes: tuple[MuxFreqFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def summary(self) -> MuxTwotoneSpectrumSummary:
        """Return native tone estimates/failures with fixed identities and owner quality."""
        summaries: list[MuxTwotoneSpectrumFitSummary | MuxSummaryFailure] = []
        for outcome in self.outcomes:
            name = None if self.names is None else self.names[outcome.index]
            element = (
                None
                if name is None
                else ElementIdentity(index=outcome.index, name=name)
            )
            if isinstance(outcome, MuxFitFailure):
                row = MuxSummaryFailure(
                    position=outcome.index,
                    name=name,
                    element=element,
                    failure=outcome.reason,
                )
            else:
                row = MuxTwotoneSpectrumFitSummary(
                    position=outcome.index,
                    name=name,
                    element=element,
                    freq=outcome.freq,
                    freq_err=outcome.freq_err,
                    fwhm=outcome.fwhm,
                    fwhm_err=outcome.fwhm_err,
                    fit_quality=fit_quality_models(
                        {"fit": outcome.quality},
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                )
            summaries.append(row)
        return MuxTwotoneSpectrumSummary(tones=summaries)

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the same owner summary for GUI and recipe consumers."""
        return self.summary().model_dump(mode="json")


def _analyze(
    req: AnalyzeRequest[MuxFreqRunResult, FreqAnalyzeParams], *, plots: Plots
) -> MuxFreqAnalyzeResult:
    """Fit the Run/Load spectra using req.analyze_params and draw them in plots."""
    params = req.analyze_params
    outcomes = MuxFreqExp().analyze(
        req.run_result,
        FreqAnalyzeOptions(model_type=params.model_type, plot_fit=params.plot_fit),
        plots=plots,
    )
    snapshot = req.run_result.cfg
    return MuxFreqAnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxFreqRunResult, MuxFreqAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose frequency, linewidth and bias-sample updates for successful fits.

    req supplies measured spectra, fit results and their original qubit
    mapping. Missing mapping returns no proposals; mismatched names raise.
    Failed fits and incomplete spectra are skipped. These are proposals only,
    using the Run/Load destinations rather than the current Form selection.
    """
    # Use the measured qubit mapping, not a later Form or alias selection.
    snapshot = req.run_result.cfg
    names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
    qubits = captured_qubits(req.ctx.parameters, names)
    if qubits is None:
        return []
    items: list[WritebackItem] = []
    for row, outcome in enumerate(req.analyze_result.outcomes):
        if (
            isinstance(outcome, MuxFitFailure)
            or not np.isfinite(req.run_result.result.signals[row]).all()
        ):
            continue
        # Only complete spectra may update the qubit calibration.
        qubit = qubits[outcome.index]
        row_items: list[WritebackItem] = []
        row_items.extend(
            [
                ParameterWriteback(
                    target_name=f"{qubit}.freq",
                    description="Qubit frequency (MHz)",
                    proposed_value=outcome.freq,
                ),
                ParameterWriteback(
                    target_name=f"{qubit}.kappa",
                    description="Qubit linewidth FWHM (MHz)",
                    proposed_value=outcome.fwhm,
                ),
                SampleWriteback(
                    coordinate_device="$mux.flux_source",
                    target_name=qubit,
                    description="Qubit frequency sample at captured bias",
                    proposed_value={
                        "freq_mhz": outcome.freq,
                        "freq_err_mhz": outcome.freq_err,
                    },
                ),
            ]
        )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def run(
    cfg: MuxFreqCfg, *, context: RunContext
) -> RunRecord[MuxFreqCfg, MuxSweepResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxFreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MuxFreqExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: FreqAnalyzeParams(),
        writeback=_get_writeback_items,
    ),
)
