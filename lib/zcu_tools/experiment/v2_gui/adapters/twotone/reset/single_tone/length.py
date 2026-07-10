from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.reset.single_tone.length import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    RoleInit,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_reset_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_scalar_float,
    reset_module_writeback_items,
    run_figure_only_analyze,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    SweepSpec,
    SweepValue,
)

SingleToneLengthRunResult: TypeAlias = LengthResult


@dataclass
class SingleToneLengthAnalyzeResult(FigureOnlyAnalyzeResult):
    # D5: the length sweep is a look-at-the-curve fit — analysis renders the decay
    # trace for the Analyze tab but extracts no scalar, so there is no writeback.
    # The single ``figure`` field is inherited from FigureOnlyAnalyzeResult.
    pass


class SingleToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        SingleToneLengthRunResult,
        SingleToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset length sweep: holds the tested "
            "pulse-reset at its calibrated sideband frequency and sweeps its "
            "duration, showing how the residual excitation decays with reset "
            "length. Runs on real hardware. Run after the reset frequency is "
            "found, to pick the shortest length that fully resets the qubit."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'reset_f' — the sideband "
            "reset frequency (= r_f - q_f) driving the tested reset (fallback "
            "0 MHz); 'q_f' / 'qub_ch' seed the tested-reset and init-pulse "
            "drive defaults."
        ),
        expects_ml=(
            "Needs a pulse-reset module (the tested reset) and a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "Proposes the ModuleLibrary module 'reset_10' — the calibrated "
            "tested single-pulse reset (carrying its md-linked sideband "
            "frequency) registered as the final reset module; the user picks "
            "the final reset length in the writeback dialog. Skipped when no "
            "cfg_snapshot is available (e.g. loaded from file)."
        ),
        recommended=(
            "A length sweep from ~0.1 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear. Allow a long relax delay between shots so the qubit "
            "fully relaxes before each reset."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # The sweep axis owns the tested-reset length
                # (set_param("length") at run, which drives waveform.length);
                # the form still shows the waveform's starting length as the
                # editable shape, mirroring the length-Rabi convention.
                "tested_reset": make_pulse_reset_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Length (us)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=30.5)
            .role("modules.tested_reset", "reset", RoleInit.INLINE)
            # The tested reset drives at the fixed sideband frequency while the
            # length is swept (notebook: freq = md.reset_f).
            .set(
                "modules.tested_reset.pulse_cfg.freq",
                md_scalar_float(ctx, "reset_f", 0.0),
            )
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", RoleInit.DISABLED)
            .role("modules.init_pulse", "pi_pulse", RoleInit.DISABLED)
            .sweep("sweep.length", SweepValue(start=0.1, stop=20.0, expts=50))
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def analyze(
        self, req: AnalyzeRequest[SingleToneLengthRunResult, NoAnalyzeParams]
    ) -> SingleToneLengthAnalyzeResult:
        return run_figure_only_analyze(LengthExp, SingleToneLengthAnalyzeResult, req)

    def get_writeback_items(
        self,
        req: WritebackRequest[SingleToneLengthRunResult, SingleToneLengthAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # D5: no scalar is fitted, so nothing is proposed back to the MetaDict.
        # Gated per-experiment 'reset_10' proposal: when md carries the calibrated
        # sideband freq, register the calibrated single-pulse reset built from this
        # run's tested_reset template (md overwrites its freq).
        items: list[WritebackItem] = []
        items.extend(
            reset_module_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                target="reset_10",
                field_md_map=[("pulse_cfg.freq", "reset_f")],
                desc="Reset with one pulse from 1 to 0",
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sidereset_length_{time.strftime('%m%d')}"
