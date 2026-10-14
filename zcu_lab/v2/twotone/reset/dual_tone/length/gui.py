from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.dual_tone.length.core import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.length.form import FORM

DualToneLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class DualToneLengthAnalyzeResult(AnalyzeResultBase):
    """Length analysis publishes a curve without fitted scalar output."""

    pass


class DualToneLengthAdapter(
    BaseAdapter[
        LengthCfg,
        DualToneLengthRunResult,
        DualToneLengthAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LengthExp.persistence_tag()
    exp_cls = LengthExp
    ExpCfg_cls = LengthCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Dual-tone reset length sweep: holds the tested two-pulse reset at "
            "its calibrated sideband frequencies and gains and sweeps its "
            "duration (both tones together), showing how the residual "
            "excitation decays with reset length. Runs on real hardware. Run "
            "last, to pick the shortest length that fully resets the qubit."
        ),
        expects_md=(
            "Tested frequencies read $qubit.ext.reset_f1 and $qubit.ext.reset_f2; gains "
            "read $qubit.ext.reset_gain1 and $qubit.ext.reset_gain2. Seeds 51 length "
            "points from 0.05 to 40.0 us, with relax delay 0.5 us. Open and Reset "
            "evaluate starting expressions once into direct input using captured sources. "
            "Inline qubit pulses read $qubit.freq and $qubit.wiring.L01.ch. Inline "
            "readout reads $resonator.freq, $resonator.wiring.ch, $resonator.wiring.ro_ch "
            "and $resonator.ext.time_of_fly + 0.05 us for trigger timing. Missing or "
            "invalid required sources leave the affected fields empty and Invalid; supply "
            "calibration or direct values before Run. No numeric calibration fallback is "
            "supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the two_pulse_reset template. Optional init "
            "pulse uses $qubit.module.x180 without fallback; an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid. $qubit.module.readout selects a "
            "captured library link, otherwise a fresh inline readout template with a "
            "const waveform. Its required parameter sources must still be present. "
            "Optional $qubit.module.reset has no fallback: an unset/dangling slot or "
            "unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "No automatic writeback. Analysis plots the length response "
            "without fitting a length. Choose the length manually; "
            "DualTonePower proposes the tested reset module."
        ),
        recommended=(
            "A length sweep from ~0.05 us to a few times the expected reset "
            "time captures the full decay; shorten the span once the plateau "
            "is clear."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> DualToneLengthRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LengthExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneLengthRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneLengthAnalyzeResult:
        LengthExp().analyze(req.run_result, None, plots=plots)
        return DualToneLengthAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_length_{time.strftime('%m%d')}"
