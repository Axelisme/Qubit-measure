from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.reset.dual_tone.length import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    md,
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
    SweepValue,
)

from ._shared import RESET_120_FIELD_MD_MAP

DualToneLengthRunResult: TypeAlias = LengthResult


@dataclass
class DualToneLengthAnalyzeResult(FigureOnlyAnalyzeResult):
    # D5: the length sweep is a look-at-the-curve fit — analysis renders the decay
    # trace for the Analyze tab but extracts no scalar, so there is no writeback.
    # The single ``figure`` field is inherited from FigureOnlyAnalyzeResult.
    pass


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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
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
                overrides={
                    "pulse1_cfg.freq": md("reset_f1", fallback=0.0),
                    "pulse2_cfg.freq": md("reset_f2", fallback=0.0),
                    "pulse1_cfg.gain": md("reset_gain1", fallback=1.0),
                    "pulse2_cfg.gain": md("reset_gain2", fallback=1.0),
                },
            )
            .readout()
            .relax_delay(0.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepValue(start=0.05, stop=40.0, expts=51),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def analyze(
        self, req: AnalyzeRequest[DualToneLengthRunResult, NoAnalyzeParams]
    ) -> DualToneLengthAnalyzeResult:
        return run_figure_only_analyze(LengthExp, DualToneLengthAnalyzeResult, req)

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
