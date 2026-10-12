from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

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
from zcu_tools.gui.cfg import (
    CfgSchema,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import path_seed
from zcu_lab.v2.twotone.reset.bath.phase.core import PhaseCfg, PhaseExp, PhaseResult

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
    ExpCfg_cls: ClassVar[Any] = PhaseCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset pi/2 phase sweep: holds the tested bath reset at its "
            "calibrated cavity freq/gain and length and sweeps the tomography "
            "pi/2 pulse phase, fitting the cosine response to find the phases "
            "that reset to ground (max) and to excited (min). Runs on real "
            "hardware. Run last, to fix the pi/2 phase of the reset module."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_bath_freq and "
            "$qubit.ext.reset_bath_gain for the tested cavity tone. Missing, "
            "null or invalid values fail cfg lowering. Other drive defaults "
            "use captured qubit/resonator frequency and wiring."
        ),
        expects_ml=(
            "Builds the tested bath-reset cfg from captured defaults and "
            "needs a readout module. Optionally references an upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="bath_reset",
                label="Tested Reset",
                shape="bath",
                locked={"pi2_cfg.phase": 0.0},
                overrides={
                    "cavity_tone_cfg.freq": path_seed(
                        "$qubit.ext.reset_bath_freq", fallback=0.0
                    ),
                    "cavity_tone_cfg.gain": path_seed(
                        "$qubit.ext.reset_bath_gain", fallback=0.1
                    ),
                },
            )
            .readout()
            .relax_delay(10.5)
            .sweep(
                "phase",
                label="Phase (deg)",
                default=SweepValue(start=-360.0, stop=360.0, expts=201),
            )
            .reps(100)
            .rounds(1000)
            .build()
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
