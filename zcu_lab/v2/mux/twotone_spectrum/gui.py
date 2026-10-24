"""GUI spectra, frequency fits and calibration proposals for mux qubits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions

from .analysis import MuxFreqFit
from .core import MuxFreqCfg, MuxFreqExp
from .form import make_form

MuxFreqRunResult: TypeAlias = RunRecord[MuxFreqCfg, MuxSweepResult]


class MuxTwotoneSpectrumFitSummary(MuxSummaryTone):
    """Decode native per-tone JSON estimates and the fitting owner diagnostics."""

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

    quality: FitQuality


class MuxTwotoneSpectrumSummary(BaseModel):
    """Decode ordered native outcomes and the complete captured name list."""

    outcomes: list[MuxTwotoneSpectrumFitSummary | MuxSummaryFailure]
    names: list[str] | None


@dataclass(frozen=True)
class MuxFreqAnalyzeResult:
    """Frequency fits or failures for each measured original tone position.

    outcomes contains per-tone estimates or failure reasons. names is the full
    qubit name list from the measured settings, or None when unavailable.
    """

    outcomes: tuple[MuxFreqFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None


class MuxFreqAdapter(
    BaseAdapter[MuxFreqCfg, MuxFreqRunResult, MuxFreqAnalyzeResult, FreqAnalyzeOptions]
):
    """Measure mux qubit spectra and propose frequency and linewidth updates."""

    sample_device_alias: ClassVar[str | None] = "$mux.flux_source"
    exp_cls = MuxFreqExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxFreqCfg]:
        """Build drive sweeps and shared readout settings from the selected mux group."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MuxFreqCfg, *, context: RunContext
    ) -> MuxFreqRunResult:
        """Measure the frequency sweeps in cfg and retain those settings with the IQ."""
        del req
        return RunRecord(cfg, MuxFreqExp().run(cfg, context=context))

    def get_analyze_params(
        self, result: MuxFreqRunResult, ctx: SessionEnv
    ) -> FreqAnalyzeOptions:
        """Seed the analysis controls without changing the core defaults."""
        del result, ctx
        return FreqAnalyzeOptions(model_type="lor", plot_fit=True)

    def analyze(
        self, req: AnalyzeRequest[MuxFreqRunResult, FreqAnalyzeOptions], *, plots: Plots
    ) -> MuxFreqAnalyzeResult:
        """Fit the Run/Load spectra using req.analyze_params and draw them in plots."""
        outcomes = MuxFreqExp().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxFreqAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxFreqRunResult, MuxFreqAnalyzeResult]
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
