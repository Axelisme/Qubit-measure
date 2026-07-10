from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.dual_tone.freq import (
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    RoleInit,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    make_two_pulse_reset_module_spec,
    md_scalar_float,
    proper_reset_freq_axis,
    reset_module_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    SweepSpec,
)

from ._shared import RESET_120_FIELD_MD_MAP

DualToneFreqRunResult: TypeAlias = FreqResult


@dataclass
class DualToneFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualToneFreqAnalyzeResult(AnalyzeResultBase):
    freq1: float
    freq2: float
    figure: Figure


class DualToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        DualToneFreqRunResult,
        DualToneFreqAnalyzeResult,
        DualToneFreqAnalyzeParams,
    ]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Dual-tone reset frequency map: a 2D sweep of the two reset-tone "
            "frequencies (pulse1 × pulse2 of the tested two-pulse reset), "
            "imaging the qubit response to locate the pair of sideband "
            "frequencies that jointly dump the excitation. Runs on real "
            "hardware in hard-sweep mode (both axes are QICK register sweeps)."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'reset_f1' / 'reset_f2' — "
            "the two sideband reset frequencies centring each sweep axis "
            "(absent → centred on 0 MHz); 'reset_gain1' / 'reset_gain2' seed "
            "the two tested-reset pulse gains; 'q_f' / 'qub_ch' seed the "
            "tested-reset and init-pulse drive defaults."
        ),
        expects_ml=(
            "Needs a two-pulse-reset module (the tested reset) and a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "Proposes the two fitted sideband frequencies into MetaDict "
            "'reset_f1' and 'reset_f2'. No ModuleLibrary writeback — "
            "registering the calibrated 'reset_120' module is left to the final "
            "length step once frequency, gain and length are all dialled in."
        ),
        recommended=(
            "Keep each axis a tight span (a few MHz) around the predicted "
            "sideband; the 2D scan cost grows with both axes. Analysis denoises "
            "the map before peak finding; wavelet smoothing is the default and "
            "Gaussian remains available for comparison."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # Each sweep axis owns one tested-reset tone frequency
                # (set_param("freq1"/"freq2") at run, which write pulse1/pulse2
                # freq); lock them so the form does not show fields the sweep
                # silently overwrites (notebook: freq=0.0).
                "tested_reset": make_two_pulse_reset_module_spec()
                .lock_literal("pulse1_cfg.freq", 0.0)
                .lock_literal("pulse2_cfg.freq", 0.0),
                "readout": make_readout_module_spec(),
            },
            sweep={
                "freq1": SweepSpec(label="Freq 1 (MHz)"),
                "freq2": SweepSpec(label="Freq 2 (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=1000, relax_delay=0.5)
            .role("modules.tested_reset", "two_pulse_reset")
            # The frequencies are swept (locked off the form); the gains are held
            # at their calibrated values so the cfg snapshot carries them forward
            # for the final reset_120 registration (D2(a) md-link).
            .set(
                "modules.tested_reset.pulse1_cfg.gain",
                md_scalar_float(ctx, "reset_gain1", 1.0),
            )
            .set(
                "modules.tested_reset.pulse2_cfg.gain",
                md_scalar_float(ctx, "reset_gain2", 1.0),
            )
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", RoleInit.DISABLED)
            .role("modules.init_pulse", "pi_pulse", RoleInit.DISABLED)
            .sweep("sweep.freq1", proper_reset_freq_axis(ctx, "reset_f1", 201))
            .sweep("sweep.freq2", proper_reset_freq_axis(ctx, "reset_f2", 201))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> DualToneFreqRunResult:
        # The dual-tone freq map runs as a 2D hard sweep (both freq axes are QICK
        # register sweeps); the notebook drives FreqExp.run with method="hard".
        # BaseAdapter.run does not pass method, so override to inject it while
        # keeping the same soc-handle / build-cfg policy.
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        soc, soccfg = require_soc_handles(req)
        return self.exp_cls().run(soc, soccfg, cfg, method="hard")

    def analyze(
        self, req: AnalyzeRequest[DualToneFreqRunResult, DualToneFreqAnalyzeParams]
    ) -> DualToneFreqAnalyzeResult:
        params = req.analyze_params
        freq1, freq2, fig = FreqExp().analyze(
            req.run_result, smooth=params.smooth, smooth_method=params.smooth_method
        )
        return DualToneFreqAnalyzeResult(freq1=freq1, freq2=freq2, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualToneFreqRunResult, DualToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="reset_f1",
                description="Dual-tone reset frequency 1 (MHz)",
                proposed_value=result.freq1,
            ),
            MetaDictWriteback(
                target_name="reset_f2",
                description="Dual-tone reset frequency 2 (MHz)",
                proposed_value=result.freq2,
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
        return f"{ctx.qub_name}_dualreset_freq_{time.strftime('%m%d')}"
