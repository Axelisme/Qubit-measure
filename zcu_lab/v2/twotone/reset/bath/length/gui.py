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

from zcu_lab.v2.twotone.reset.bath.length.core import LengthCfg, LengthExp, LengthResult
from zcu_lab.v2.twotone.reset.bath.length.form import make_form

BathLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class BathLengthAnalyzeResult(AnalyzeResultBase):
    """The length analysis publishes a figure without fitted scalars."""


class BathLengthAdapter(
    BaseAdapter[
        LengthCfg,
        BathLengthRunResult,
        BathLengthAnalyzeResult,
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
            "Bath-reset length sweep: holds the tested bath reset at its "
            "calibrated cavity frequency and gain and sweeps its duration "
            "(cavity and qubit tones together), with an internal 4-point pi/2 "
            "tomography phase axis, showing how the residual excitation decays "
            "with reset length. Runs on real hardware. Run after the cavity "
            "freq/gain are found, to pick the shortest fully-resetting length."
        ),
        expects_md=(
            "Tested cavity frequency and gain read $qubit.ext.reset_bath_freq and "
            "$qubit.ext.reset_bath_gain. Seeds 201 length points from 0.05 to 15.0 us, "
            "with relax delay 10.5 us. Run replaces the pi/2 phase with its tomography "
            "cycle. Open and Reset evaluate starting expressions once into direct input "
            "using captured sources. Inline qubit pulses read $qubit.freq and "
            "$qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the bath_reset template. Optional init pulse "
            "uses $qubit.module.x180 without fallback; an unset/dangling slot or unbound "
            "alias leaves it disabled and Valid. The pi/2 tone uses $qubit.module.x90, "
            "otherwise a fresh pi2_pulse template. $qubit.module.readout selects a "
            "captured library link, otherwise a fresh inline readout template with a "
            "const waveform. Its required parameter sources must still be present. "
            "Optional $qubit.module.reset has no fallback: an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid."
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

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: LengthCfg,
        *,
        context: RunContext,
    ) -> BathLengthRunResult:
        del req
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> BathLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return BathLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_bathreset_len_{time.strftime('%m%d')}"
