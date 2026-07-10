from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.dual_tone.power import (
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    md,
    reset_module_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    SweepValue,
)

from ._shared import RESET_120_FIELD_MD_MAP

DualTonePowerRunResult: TypeAlias = PowerResult


@dataclass
class DualTonePowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
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
            "Sweep each gain across its full usable range. Analysis denoises "
            "the map before picking the optimum; wavelet smoothing is the "
            "default and Gaussian remains available for comparison. The "
            "frequencies are held at 'reset_f1' / 'reset_f2', so calibrate "
            "those first."
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
                role_id="two_pulse_reset",
                label="Tested Reset",
                shape="two_pulse",
                locked={"pulse1_cfg.gain": 0.0, "pulse2_cfg.gain": 0.0},
                overrides={
                    "pulse1_cfg.freq": md("reset_f1", fallback=0.0),
                    "pulse2_cfg.freq": md("reset_f2", fallback=0.0),
                },
            )
            .readout()
            .relax_delay(0.5)
            .sweep(
                "gain1",
                label="Gain 1 (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .sweep(
                "gain2",
                label="Gain 2 (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[DualTonePowerRunResult, DualTonePowerAnalyzeParams]
    ) -> DualTonePowerAnalyzeResult:
        params = req.analyze_params
        gain1, gain2, fig = PowerExp().analyze(
            req.run_result, smooth=params.smooth, smooth_method=params.smooth_method
        )
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
