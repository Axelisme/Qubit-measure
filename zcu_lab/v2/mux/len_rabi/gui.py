"""Mux len_rabi source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.mux_results import captured_qubits, indexed_summary
from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAnalyzeParams

from .analysis import MuxLenRabiFit
from .core import MuxLenRabiCfg, MuxLenRabiExp

MuxLenRabiRunResult: TypeAlias = RunRecord[MuxLenRabiCfg, MuxSweepResult]


@dataclass(frozen=True)
class MuxLenRabiAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxLenRabiFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present original positions/names, numerical values, quality or failure."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                values = {
                    "pi_len": outcome.pi_len,
                    "pi_len_err": outcome.pi_len_err,
                    "pi2_len": outcome.pi2_len,
                    "pi2_len_err": outcome.pi2_len_err,
                    "rabi_f": outcome.rabi_f,
                    "rabi_f_err": outcome.rabi_f_err,
                    "fit_quality": fit_quality_summary(
                        {"fit": outcome.quality},
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


class MuxLenRabiAdapter(
    BaseAdapter[
        MuxLenRabiCfg,
        MuxLenRabiRunResult,
        MuxLenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxLenRabiExp.persistence_tag()
    exp_cls = MuxLenRabiExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self, req: RunRequest, cfg: MuxLenRabiCfg, *, context: RunContext
    ) -> MuxLenRabiRunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxLenRabiExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxLenRabiRunResult, LenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxLenRabiAnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxLenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(decay=params.decay, fit_phase=params.fit_phase),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxLenRabiAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxLenRabiRunResult, MuxLenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose only complete, successful tones using the same Run/Load capture.

        Missing capture returns no items. Captured bindings must match any source
        cfg names/count; errors propagate. Partial/nonfinite rows and failed fits
        produce no calibration items. Pulse proposals copy source cfg only, never
        current library values. This method neither writes nor applies proposals.
        """
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
            qubit = qubits[outcome.index]
            items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.ext.rabi_freq",
                    description="Rabi oscillation frequency (MHz)",
                    proposed_value=outcome.rabi_f,
                )
            )
            if snapshot is None:
                continue
            pulse = snapshot.tones[outcome.index].drive
            parameters = req.ctx.parameters
            if parameters is None:
                raise ValueError("Rabi pulse proposals require captured parameters")
            gain = pulse.gain
            if not isinstance(gain, (int, float)):
                raise ValueError("Length Rabi pi_product requires scalar source gain")
            for suffix, slot, value, description in (
                ("pi_amp", "x180", outcome.pi_len, "Pi pulse"),
                ("pi2_amp", "x90", outcome.pi2_len, "Pi/2 pulse"),
            ):
                target = f"{qubit}.pulses.{suffix}"
                items.append(
                    module_writeback(
                        pulse,
                        target=target,
                        description=description,
                        field_updates=(("waveform.length", value),),
                    )
                )
                items.append(
                    ParameterWriteback(
                        target_name=f"{qubit}.module.{slot}",
                        description=f"{description} slot",
                        proposed_value=parameters.library_key(target),
                    )
                )
            items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.ext.pi_product",
                    description="Pi pulse gain times length (a.u. * us)",
                    proposed_value=gain * outcome.pi_len,
                )
            )
        return items
