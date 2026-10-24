from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.reset.dual_tone.power.core import (
    PowerAnalysis,
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.power.form import make_form

DualTonePowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class DualTonePowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


class DualTonePowerAdapter(
    BaseAdapter[
        PowerCfg,
        DualTonePowerRunResult,
        PowerAnalysis,
        DualTonePowerAnalyzeParams,
    ]
):
    exp_cls = PowerExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> DualTonePowerRunResult:
        del req
        result = PowerExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualTonePowerRunResult, DualTonePowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> PowerAnalysis:
        params = req.analyze_params
        return PowerExp().analyze(
            req.run_result,
            PowerAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )

    def get_writeback_items(
        self,
        req: WritebackRequest[DualTonePowerRunResult, PowerAnalysis],
    ) -> Sequence[WritebackItem]:
        """Propose ext gains and the Run-tested reset with those fitted gains.

        None cfg or capture retains only scalars. Capture resolution and
        materialization failures propagate; tested frequencies stay unchanged.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_gain1",
                description="Dual-tone reset gain 1 (a.u.)",
                proposed_value=result.gain1,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_gain2",
                description="Dual-tone reset gain 2 (a.u.)",
                proposed_value=result.gain2,
            ),
        ]
        cfg = req.run_result.cfg
        parameters = req.ctx.parameters
        if cfg is None or parameters is None:
            return items

        root = parameters.resolve_path("$qubit.reset.reset_120")
        items.append(
            module_writeback(
                cfg.tested_reset,
                target="$qubit.reset.reset_120",
                description="Reset with two pulses from 1 to 2 to 0",
                field_updates=(
                    ("pulse1_cfg.gain", result.gain1),
                    ("pulse2_cfg.gain", result.gain2),
                ),
            )
        )
        items.append(
            ParameterWriteback(
                target_name="$qubit.module.reset",
                description="Select dual-tone reset for downstream experiments",
                proposed_value=root,
            )
        )
        return items
