from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.reset.bath.phase.core import PhaseCfg, PhaseExp, PhaseResult
from zcu_lab.v2.twotone.reset.bath.phase.form import make_form

BathPhaseRunResult: TypeAlias = RunRecord[PhaseCfg, PhaseResult]


@dataclass
class BathPhaseAnalyzeResult(AnalyzeResultBase):
    max_phase: float
    min_phase: float


class BathPhaseAdapter(
    BaseAdapter[
        PhaseCfg,
        BathPhaseRunResult,
        BathPhaseAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = PhaseExp.persistence_tag()
    exp_cls = PhaseExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PhaseCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: PhaseCfg,
        *,
        context: RunContext,
    ) -> BathPhaseRunResult:
        del req
        result = PhaseExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathPhaseRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> BathPhaseAnalyzeResult:
        analysis = PhaseExp().analyze(req.run_result, None, plots=plots)
        return BathPhaseAnalyzeResult(
            max_phase=analysis.max_phase, min_phase=analysis.min_phase
        )

    def get_writeback_items(
        self,
        req: WritebackRequest[BathPhaseRunResult, BathPhaseAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose ext phases, independent tested variants and a ground slot.

        None cfg or capture retains only scalars. Each variant copies this
        Run's tested_reset and changes only pi2_cfg.phase. Capture resolution
        and materialization failures propagate; ext is not reread.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_max_phase",
                description="Bath-reset pi/2 phase for ground reset (deg)",
                proposed_value=result.max_phase,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_min_phase",
                description="Bath-reset pi/2 phase for excited reset (deg)",
                proposed_value=result.min_phase,
            ),
        ]

        cfg = req.run_result.cfg
        parameters = req.ctx.parameters
        if cfg is None or parameters is None:
            return items

        ground_root = parameters.resolve_path("$qubit.reset.reset_bath")
        for target, phase, description in (
            ("$qubit.reset.reset_bath", result.max_phase, "Bath reset to ground"),
            ("$qubit.reset.reset_bath_e", result.min_phase, "Bath reset to excited"),
        ):
            items.append(
                module_writeback(
                    cfg.modules.tested_reset,
                    target=target,
                    description=description,
                    field_updates=(("pi2_cfg.phase", phase),),
                )
            )
        items.append(
            ParameterWriteback(
                target_name="$qubit.module.reset",
                description="Select bath reset for downstream experiments",
                proposed_value=ground_root,
            )
        )

        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_bathreset_phase_{time.strftime('%m%d')}"
