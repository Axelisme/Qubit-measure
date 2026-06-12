from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.bath.phase import (
    PhaseCfg,
    PhaseExp,
    PhaseResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_bath_reset_module_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_scalar_float,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ModuleWriteback,
    NoAnalyzeParams,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value

BathPhaseRunResult: TypeAlias = PhaseResult


@dataclass
class BathPhaseAnalyzeResult(AnalyzeResultBase):
    max_phase: float
    min_phase: float
    figure: Figure


class BathPhaseAdapter(
    BaseAdapter[
        PhaseCfg,
        BathPhaseRunResult,
        BathPhaseAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = PhaseExp
    ExpCfg_cls: ClassVar[Any] = PhaseCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Bath-reset pi/2 phase sweep: holds the tested bath reset at its "
                "calibrated cavity freq/gain and length and sweeps the tomography "
                "pi/2 pulse phase, fitting the cosine response to find the phases "
                "that reset to ground (max) and to excited (min). Runs on real "
                "hardware. Run last, to fix the pi/2 phase of the reset module."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'bathreset_freq' / "
                "'bathreset_gain' — the calibrated cavity frequency and gain so the "
                "cfg snapshot carries the fully calibrated reset forward for "
                "registration; 'q_f' / 'qub_ch' / 'res_ch' seed the bath-reset "
                "tone drive defaults."
            ),
            expects_ml=(
                "Needs a bath-reset module (the tested reset) and a readout "
                "module. Optionally references a calibrated upstream reset and an "
                "init pulse (a library pi pulse when present) — both disabled when "
                "no library entry exists."
            ),
            typical_writeback=(
                "Proposes the ground-reset phase into MetaDict "
                "'bathreset_max_phase' and the excited-reset phase into "
                "'bathreset_min_phase'. Also proposes two ModuleLibrary modules: "
                "'reset_bath' (the calibrated tested reset with pi/2 phase set to "
                "the max/ground phase) and 'reset_bath_e' (pi/2 phase set to the "
                "min/excited phase) — both skipped when no cfg_snapshot is "
                "available (e.g. loaded from file) (D2(a))."
            ),
            recommended=(
                "A full -360..360 deg sweep captures a clean cosine; the fit picks "
                "the max / min phases automatically. Allow a long relax delay so "
                "the qubit fully relaxes before each shot."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # The sweep axis owns the pi/2 phase (set_param("pi2_phase") at
                # run); lock it off the form so it never shows a field the sweep
                # silently overwrites.
                "tested_reset": make_bath_reset_module_spec().lock_literal(
                    "pi2_cfg.phase", 0.0
                ),
                "readout": make_readout_module_spec(),
            },
            sweep={"phase": SweepSpec(label="Phase (deg)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=1000, relax_delay=10.5)
            .role("modules.tested_reset", "bath_reset")
            # Hold the cavity tone at its calibrated freq/gain while the phase is
            # swept, so the cfg snapshot carries the calibrated reset forward for
            # the final reset_bath / reset_bath_e registration (D2(a) md-link).
            .set(
                "modules.tested_reset.cavity_tone_cfg.freq",
                md_scalar_float(ctx, "bathreset_freq", 0.0),
            )
            .set(
                "modules.tested_reset.cavity_tone_cfg.gain",
                md_scalar_float(ctx, "bathreset_gain", 0.1),
            )
            .role("modules.readout", "readout", prefer_blank=True)
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.phase", SweepValue(start=-360.0, stop=360.0, expts=201))
            .build()
        )

    def get_analyze_params(
        self, result: BathPhaseRunResult, ctx: ExpContext
    ) -> NoAnalyzeParams:
        return NoAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[BathPhaseRunResult, NoAnalyzeParams]
    ) -> BathPhaseAnalyzeResult:
        max_phase, min_phase, fig = PhaseExp().analyze(req.run_result)
        return BathPhaseAnalyzeResult(
            max_phase=max_phase, min_phase=min_phase, figure=fig
        )

    def get_writeback_items(
        self,
        req: WritebackRequest[BathPhaseRunResult, BathPhaseAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="bathreset_max_phase",
                description="Bath-reset pi/2 phase for ground reset (deg)",
                proposed_value=result.max_phase,
            ),
            MetaDictWriteback(
                target_name="bathreset_min_phase",
                description="Bath-reset pi/2 phase for excited reset (deg)",
                proposed_value=result.min_phase,
            ),
        ]

        # D2(a): as the last bath step, register two final reset modules from the
        # calibrated tested_reset (which already carries its md-linked cavity
        # freq/gain). Each is the same reset with only pi2_cfg.phase overridden —
        # 'reset_bath' to the max phase (resets to ground), 'reset_bath_e' to the
        # min phase (resets to excited). Skipped without a cfg_snapshot.
        snapshot = req.run_result.cfg_snapshot
        if snapshot is not None:
            for target, phase, desc in [
                (
                    "reset_bath",
                    result.max_phase,
                    "Reset to Ground with cavity-assisted bath reset",
                ),
                (
                    "reset_bath_e",
                    result.min_phase,
                    "Reset to Excited with cavity-assisted bath reset",
                ),
            ]:
                spec, value = module_cfg_to_value(snapshot.modules.tested_reset)
                value.with_field("pi2_cfg.phase", phase)
                items.append(
                    ModuleWriteback(
                        target_name=target,
                        description=desc,
                        edit_schema=CfgSchema(spec=spec, value=value),
                    )
                )

        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_bathreset_phase_{time.strftime('%m%d')}"
