from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.single_tone.length.core import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.reset.single_tone.length.form import make_form

SingleToneLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class SingleToneLengthAnalyzeResult(AnalyzeResultBase):
    """Length analysis publishes a curve without fitted scalar output."""

    pass


class SingleToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        SingleToneLengthRunResult,
        SingleToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LengthExp.persistence_tag()
    exp_cls = LengthExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[LengthCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset length sweep: holds the tested "
            "pulse-reset at its calibrated sideband frequency and sweeps its "
            "duration, showing how the residual excitation decays with reset "
            "length. Runs on real hardware. Run after the reset frequency is "
            "found, to pick the shortest length that fully resets the qubit."
        ),
        expects_md=(
            "The tested pulse frequency reads $qubit.ext.reset_f. Seeds 50 length points "
            "from 0.1 to 20.0 us, with relax delay 30.5 us. Open and Reset evaluate "
            "starting expressions once into direct input using captured sources. Inline "
            "qubit pulses read $qubit.freq and $qubit.wiring.L01.ch. Inline readout reads "
            "$resonator.freq, $resonator.wiring.ch, $resonator.wiring.ro_ch and "
            "$resonator.ext.time_of_fly + 0.05 us for trigger timing. Missing or invalid "
            "required sources leave the affected fields empty and Invalid; supply "
            "calibration or direct values before Run. No numeric calibration fallback is "
            "supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the reset template. Optional init pulse uses "
            "$qubit.module.x180 without fallback; an unset/dangling slot or unbound alias "
            "leaves it disabled and Valid. $qubit.module.readout selects a captured "
            "library link, otherwise a fresh inline readout template with a const "
            "waveform. Its required parameter sources must still be present. Optional "
            "$qubit.module.reset has no fallback: an unset/dangling slot or unbound alias "
            "leaves it disabled and Valid."
        ),
        typical_writeback=(
            "No automatic writeback. Analysis plots the length response "
            "without fitting a length. Choose the length manually; "
            "SingleToneFreq proposes the tested reset module."
        ),
        recommended=(
            "A length sweep from ~0.1 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear. Allow a long relax delay between shots so the qubit "
            "fully relaxes before each reset."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: LengthCfg,
        *,
        context: RunContext,
    ) -> SingleToneLengthRunResult:
        del req
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SingleToneLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SingleToneLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return SingleToneLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sidereset_length_{time.strftime('%m%d')}"
