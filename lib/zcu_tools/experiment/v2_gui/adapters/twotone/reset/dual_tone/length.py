from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.dual_tone.length import (
    LengthCfg,
    LengthExp,
    LengthResult,
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
    NoAnalyzeParams,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

from ._shared import RESET_120_FIELD_MD_MAP

DualToneLengthRunResult: TypeAlias = LengthResult


@dataclass
class DualToneLengthAnalyzeResult(AnalyzeResultBase):
    # D5: the length sweep is a look-at-the-curve fit — analysis renders the
    # decay trace for the Analyze tab but extracts no scalar, so there is no
    # writeback. Only the figure is carried.
    figure: Figure


class DualToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        DualToneLengthRunResult,
        DualToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Dual-tone reset length sweep: holds the tested two-pulse reset at "
                "its calibrated sideband frequencies and gains and sweeps its "
                "duration (both tones together), showing how the residual "
                "excitation decays with reset length. Runs on real hardware. Run "
                "last, to pick the shortest length that fully resets the qubit."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'reset_f1' / 'reset_f2' "
                "and 'reset_gain1' / 'reset_gain2' — the calibrated sideband "
                "frequencies and gains driving the tested reset (so the cfg "
                "snapshot carries the fully calibrated reset); 'q_f' / 'qub_ch' "
                "seed the tested-reset and init-pulse drive defaults."
            ),
            expects_ml=(
                "Needs a two-pulse-reset module (the tested reset) and a readout "
                "module. Optionally references a calibrated upstream reset and an "
                "init pulse (a library pi pulse when present) — both disabled when "
                "no library entry exists."
            ),
            typical_writeback=(
                "Proposes the ModuleLibrary module 'reset_120' — the calibrated "
                "tested two-pulse reset (carrying its md-linked sideband "
                "frequencies and gains) registered as the final reset module; the "
                "user picks the final reset length in the writeback dialog. "
                "Skipped when no cfg_snapshot is available (e.g. loaded from file)."
            ),
            recommended=(
                "A length sweep from ~0.05 us to a few times the expected reset "
                "time captures the full decay; shorten the span once the plateau "
                "is clear."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # The sweep axis owns both tested-reset tone lengths
                # (set_param("length") at run, which drives both waveforms); the
                # form still shows the waveforms' starting lengths as the editable
                # shape, mirroring the length-Rabi convention.
                "tested_reset": make_two_pulse_reset_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Length (us)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=0.5)
            .role("modules.tested_reset", "two_pulse_reset")
            # Hold both tones at their calibrated frequencies and gains while the
            # length is swept, so the cfg snapshot carries the fully calibrated
            # reset for the final reset_120 registration (D2(a) md-link).
            .set(
                "modules.tested_reset.pulse1_cfg.freq",
                md_scalar_float(ctx, "reset_f1", 0.0),
            )
            .set(
                "modules.tested_reset.pulse2_cfg.freq",
                md_scalar_float(ctx, "reset_f2", 0.0),
            )
            .set(
                "modules.tested_reset.pulse1_cfg.gain",
                md_scalar_float(ctx, "reset_gain1", 1.0),
            )
            .set(
                "modules.tested_reset.pulse2_cfg.gain",
                md_scalar_float(ctx, "reset_gain2", 1.0),
            )
            .role("modules.readout", "readout", prefer_blank=True)
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.length", SweepValue(start=0.05, stop=40.0, expts=51))
            .build()
        )

    def get_analyze_params(
        self, result: DualToneLengthRunResult, ctx: ExpContext
    ) -> NoAnalyzeParams:
        return NoAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[DualToneLengthRunResult, NoAnalyzeParams]
    ) -> DualToneLengthAnalyzeResult:
        fig = LengthExp().analyze(req.run_result)
        return DualToneLengthAnalyzeResult(figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualToneLengthRunResult, DualToneLengthAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # D5: no scalar is fitted, so nothing is proposed back to the MetaDict.
        # Gated per-experiment 'reset_120' proposal: when md carries the calibrated
        # freqs + gains, register the calibrated two-pulse reset built from this
        # run's tested_reset template (md overwrites those fields).
        items: list[WritebackItem] = []
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
        return f"{ctx.qub_name}_dualreset_length_{time.strftime('%m%d')}"
