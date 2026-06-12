from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.dual_tone.power import (
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    make_two_pulse_reset_module_spec,
    md_scalar_float,
    reset_module_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

from ._shared import RESET_120_FIELD_MD_MAP

DualTonePowerRunResult: TypeAlias = PowerResult


@dataclass
class DualTonePowerAnalyzeParams:
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=2)]


@dataclass
class DualTonePowerAnalyzeResult(AnalyzeResultBase):
    gain1: float
    gain2: float
    figure: Figure


class DualTonePowerAdapter(
    BaseAdapter[
        PowerCfg,
        DualTonePowerRunResult,
        DualTonePowerAnalyzeResult,
        DualTonePowerAnalyzeParams,
    ]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Dual-tone reset gain map: a 2D sweep of the two reset-tone gains "
                "(pulse1 × pulse2 of the tested two-pulse reset) at the calibrated "
                "sideband frequencies, imaging the residual excitation to pick the "
                "gain pair that resets most completely. Runs on real hardware."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'reset_f1' / 'reset_f2' — "
                "the two sideband frequencies held fixed on the tested reset; "
                "'q_f' / 'qub_ch' seed the tested-reset and init-pulse drive "
                "defaults."
            ),
            expects_ml=(
                "Needs a two-pulse-reset module (the tested reset) and a readout "
                "module. Optionally references a calibrated upstream reset and an "
                "init pulse (a library pi pulse when present) — both disabled when "
                "no library entry exists."
            ),
            typical_writeback=(
                "Proposes the two best gains into MetaDict 'reset_gain1' and "
                "'reset_gain2'. No ModuleLibrary writeback — these seed the final "
                "'reset_120' registration done at the length step (D2(a))."
            ),
            recommended=(
                "Sweep each gain across its full usable range; 'smooth' "
                "Gaussian-filters the map before picking the optimum (sigma ~1 by "
                "default). The frequencies are held at 'reset_f1' / 'reset_f2', so "
                "calibrate those first."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # Each sweep axis owns one tested-reset tone gain
                # (set_param("gain1"/"gain2") at run, which write pulse1/pulse2
                # gain); lock them off the form.
                "tested_reset": make_two_pulse_reset_module_spec()
                .lock_literal("pulse1_cfg.gain", 0.0)
                .lock_literal("pulse2_cfg.gain", 0.0),
                "readout": make_readout_module_spec(),
            },
            sweep={
                "gain1": SweepSpec(label="Gain 1 (a.u.)"),
                "gain2": SweepSpec(label="Gain 2 (a.u.)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=0.5)
            .role("modules.tested_reset", "two_pulse_reset")
            # The gains are swept (locked off the form); the frequencies are held
            # at their calibrated values so the cfg snapshot carries them forward
            # for the final reset_120 registration (D2(a) md-link).
            .set(
                "modules.tested_reset.pulse1_cfg.freq",
                md_scalar_float(ctx, "reset_f1", 0.0),
            )
            .set(
                "modules.tested_reset.pulse2_cfg.freq",
                md_scalar_float(ctx, "reset_f2", 0.0),
            )
            .role("modules.readout", "readout", prefer_blank=True)
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .sweep("sweep.gain1", 0.0, 1.0, 51)
            .sweep("sweep.gain2", 0.0, 1.0, 51)
            .build()
        )

    def get_analyze_params(
        self, result: DualTonePowerRunResult, ctx: ExpContext
    ) -> DualTonePowerAnalyzeParams:
        return DualTonePowerAnalyzeParams(smooth=1.0)

    def analyze(
        self, req: AnalyzeRequest[DualTonePowerRunResult, DualTonePowerAnalyzeParams]
    ) -> DualTonePowerAnalyzeResult:
        params = req.analyze_params
        gain1, gain2, fig = PowerExp().analyze(req.run_result, smooth=params.smooth)
        return DualTonePowerAnalyzeResult(gain1=gain1, gain2=gain2, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualTonePowerRunResult, DualTonePowerAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="reset_gain1",
                description="Dual-tone reset gain 1 (a.u.)",
                proposed_value=result.gain1,
            ),
            MetaDictWriteback(
                target_name="reset_gain2",
                description="Dual-tone reset gain 2 (a.u.)",
                proposed_value=result.gain2,
            ),
        ]
        items.extend(
            reset_module_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                target="reset_120",
                field_md_map=RESET_120_FIELD_MD_MAP,
                desc="Reset with two pulse from 1 to 2 to 0",
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_dualreset_gain_{time.strftime('%m%d')}"
