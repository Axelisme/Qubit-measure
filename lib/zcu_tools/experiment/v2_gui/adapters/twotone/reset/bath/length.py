from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.reset.bath.length import (
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

from ._shared import bath_reset_writeback_items

BathLengthRunResult: TypeAlias = LengthResult

# The domain replaces pi2_cfg.phase with a 4-point QickSweep1D("phase", 0, 270)
# for the tomography readout, so the form value is inert; lock it to the
# notebook offset to keep the form from showing a field the run overwrites.
_PI2_PHASE_OFFSET_DEG: float = 90.0


@dataclass
class BathLengthAnalyzeResult(FigureOnlyAnalyzeResult):
    # D5: the length sweep is a look-at-the-curve fit — analysis renders the decay
    # trace for the Analyze tab but extracts no scalar, so there is no writeback.
    # The single ``figure`` field is inherited from FigureOnlyAnalyzeResult.
    pass


class BathLengthAdapter(
    BaseAdapter[
        LengthCfg,
        BathLengthRunResult,
        BathLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset length sweep: holds the tested bath reset at its "
            "calibrated cavity frequency and gain and sweeps its duration "
            "(cavity and qubit tones together), with an internal 4-point pi/2 "
            "tomography phase axis, showing how the residual excitation decays "
            "with reset length. Runs on real hardware. Run after the cavity "
            "freq/gain are found, to pick the shortest fully-resetting length."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'bathreset_freq' / "
            "'bathreset_gain' — the calibrated cavity frequency and gain "
            "driving the tested reset (so the cfg snapshot carries them "
            "forward); 'q_f' / 'qub_ch' / 'res_ch' seed the bath-reset and "
            "init-pulse drive defaults."
        ),
        expects_ml=(
            "Needs a bath-reset module (the tested reset) and a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "No writeback — the chosen reset length is read off the decay "
            "curve by eye, and registering the calibrated 'reset_bath' module "
            "is left to the final phase step."
        ),
        recommended=(
            "A length sweep from ~0.05 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear. Allow a long relax delay so the qubit fully relaxes "
            "before each reset."
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
                locked={"pi2_cfg.phase": _PI2_PHASE_OFFSET_DEG},
                overrides={
                    "cavity_tone_cfg.freq": md("bathreset_freq", fallback=0.0),
                    "cavity_tone_cfg.gain": md("bathreset_gain", fallback=0.1),
                },
            )
            .readout()
            .relax_delay(10.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepValue(start=0.05, stop=15.0, expts=201),
            )
            .reps(100)
            .rounds(1000)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def analyze(
        self, req: AnalyzeRequest[BathLengthRunResult, NoAnalyzeParams]
    ) -> BathLengthAnalyzeResult:
        return run_figure_only_analyze(LengthExp, BathLengthAnalyzeResult, req)

    def get_writeback_items(
        self,
        req: WritebackRequest[BathLengthRunResult, BathLengthAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # D5: no scalar is fitted, so nothing is proposed back to the MetaDict.
        # Gated per-experiment 'reset_bath' / 'reset_bath_e' proposals: emitted when
        # md carries the calibrated cavity freq/gain and the matching pi/2 phase.
        items: list[WritebackItem] = []
        items.extend(bath_reset_writeback_items(req.ctx, req.run_result.cfg_snapshot))
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_bathreset_len_{time.strftime('%m%d')}"
