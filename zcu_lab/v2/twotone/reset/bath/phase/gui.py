from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.bath.phase.core import PhaseCfg, PhaseExp, PhaseResult
from zcu_lab.v2.twotone.reset.bath.phase.form import FORM

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
    ExpCfg_cls = PhaseCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset pi/2 phase sweep: holds the tested bath reset at its "
            "calibrated cavity freq/gain and length and sweeps the tomography "
            "pi/2 pulse phase, fitting the cosine response to find the phases "
            "that reset to ground (max) and to excited (min). Runs on real "
            "hardware. Run last, to fix the pi/2 phase of the reset module."
        ),
        expects_md=(
            "Tested cavity frequency and gain read $qubit.ext.reset_bath_freq and "
            "$qubit.ext.reset_bath_gain. Seeds 201 phase points from -360 to 360 degrees, "
            "with relax delay 10.5 us. Open and Reset evaluate starting expressions once "
            "into direct input using captured sources. Inline qubit pulses read "
            "$qubit.freq and $qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the bath_reset template. Optional init pulse "
            "uses $qubit.module.x180 without fallback; an unset/dangling slot or unbound "
            "alias leaves it disabled and Valid. The pi/2 tone uses $qubit.module.x90, "
            "otherwise a fresh pi2_pulse template. $qubit.module.readout selects a "
            "captured library link, otherwise a fresh inline readout template with a "
            "const waveform. Its required parameter sources must still be present. "
            "Optional $qubit.module.reset has no fallback: an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes phases to $qubit.ext.reset_bath_max_phase and "
            "$qubit.ext.reset_bath_min_phase. With Run cfg and capture, "
            "proposes $qubit.reset.reset_bath and $qubit.reset.reset_bath_e "
            "from tested_reset, replacing only pi/2 phase with max/ground "
            "and min/excited respectively. Tested cavity frequency, gain "
            "and all other fields remain. An optional $qubit.module.reset "
            "row selects ground by default; it can be edited to excited."
        ),
        recommended=(
            "A full -360..360 deg sweep captures a clean cosine; the fit picks "
            "the max / min phases automatically. Allow a long relax delay so "
            "the qubit fully relaxes before each shot."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> BathPhaseRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
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
            spec, value = module_cfg_to_value(cfg.modules.tested_reset)
            value.with_field("pi2_cfg.phase", phase)
            items.append(
                ModuleWriteback(
                    target_name=target,
                    description=description,
                    edit_schema=CfgSchema(spec=spec, value=value),
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
