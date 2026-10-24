from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.reset.single_tone.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.twotone.reset.single_tone.freq.form import make_form

SingleToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class SingleToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float


class SingleToneFreqAdapter(
    BaseAdapter[FreqCfg, SingleToneFreqRunResult, SingleToneFreqAnalyzeResult]
):
    exp_cls = FreqExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: None (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> SingleToneFreqRunResult:
        del req
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SingleToneFreqRunResult, None],
        *,
        plots: Plots,
    ) -> SingleToneFreqAnalyzeResult:
        result = FreqExp().analyze(req.run_result, None, plots=plots)
        return SingleToneFreqAnalyzeResult(freq=result.freq, fwhm=result.fwhm)

    def get_writeback_items(
        self,
        req: WritebackRequest[SingleToneFreqRunResult, SingleToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose ext scalars and the Run-tested reset with fitted frequency.

        None cfg or capture retains only scalars. Capture resolution and
        materialization failures propagate; no calibration is reread.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_f",
                description="Sideband reset frequency (MHz)",
                proposed_value=result.freq,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_fwhm",
                description="Reset sideband linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]
        cfg = req.run_result.cfg
        parameters = req.ctx.parameters
        if cfg is None or parameters is None:
            return items

        root = parameters.resolve_path("$qubit.reset.reset_10")
        items.append(
            module_writeback(
                cfg.tested_reset,
                target="$qubit.reset.reset_10",
                description="Reset with one pulse from 1 to 0",
                field_updates=(("pulse_cfg.freq", result.freq),),
            )
        )
        items.append(
            ParameterWriteback(
                target_name="$qubit.module.reset",
                description="Select single-tone reset for downstream experiments",
                proposed_value=root,
            )
        )
        return items
