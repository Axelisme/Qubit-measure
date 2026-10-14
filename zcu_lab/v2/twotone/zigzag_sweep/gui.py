"""GUI adapters for gain and frequency scans of the ZigZag experiment."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.zigzag_sweep.core import (
    ZigZagScanAnalyzeOptions,
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanResult,
)

from .form import FREQ_FORM, GAIN_FORM

ZigZagScanRunResult: TypeAlias = RunRecord[ZigZagScanCfg, ZigZagScanResult]


@dataclass
class ZigZagScanAnalyzeParams:
    find_min: Annotated[float | None, ParamMeta(label="Search from")] = None
    find_max: Annotated[float | None, ParamMeta(label="Search to")] = None


@dataclass
class ZigZagScanAnalyzeResult(AnalyzeResultBase):
    best_value: float


class _ZigZagScanBase(
    BaseAdapter[
        ZigZagScanCfg,
        ZigZagScanRunResult,
        ZigZagScanAnalyzeResult,
        ZigZagScanAnalyzeParams,
    ]
):
    """Shared body; subclasses choose the single swept parameter."""

    exp_cls = ZigZagScanExp
    ExpCfg_cls: ClassVar[Any] = ZigZagScanCfg

    sweep_key: ClassVar[str]

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> ZigZagScanRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ZigZagScanExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[ZigZagScanRunResult, ZigZagScanAnalyzeParams],
        *,
        plots: Plots,
    ) -> ZigZagScanAnalyzeResult:
        params = req.analyze_params
        analysis = ZigZagScanExp().analyze(
            req.run_result,
            ZigZagScanAnalyzeOptions(find_range=(params.find_min, params.find_max)),
            plots=plots,
        )
        return ZigZagScanAnalyzeResult(best_value=analysis.min_value)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_zigzag_scan_{self.sweep_key}_{time.strftime('%m%d')}"


_SCAN_RECOMMENDED = (
    "Analysis picks the sweep value whose signal changes least across "
    "repetitions (smoothed sum of step differences). Set 'Search from' / "
    "'Search to' to ignore edges of the sweep. Keep n_times small (~6); the "
    "sweep multiplies the run time."
)


class ZigZagScanGainAdapter(_ZigZagScanBase):
    form = GAIN_FORM
    experiment_tag: ClassVar[str | None] = ZigZagScanExp.persistence_tag()
    sweep_key: ClassVar[str] = "gain"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Zig-zag gain scan: runs the zig-zag repetition sequence at each gain of "
            "the repeated pulse, and picks the gain where the signal stays flattest "
            "across repetitions. Runs on real hardware."
        ),
        expects_md=(
            "The gain scan runs 0.8 to 1.2 * qubit.module.x180.gain; for X90 repetition "
            "adjust the window to x90. The frequency scan runs qubit.freq ± 2 MHz. Both "
            "use 101 points and maximum repetitions 6. Relax delay is 5 * qubit.t1 us. "
            "Missing calibration leaves the affected fields empty and Invalid; no "
            "numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). Drive slots "
            "use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. "
            "Unset or dangling slots use the named template, not a legacy blank. Key "
            "choices remain available. Inline templates read qubit.freq, "
            "qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and "
            "resonator.ext.time_of_fly + 0.05 us; missing required calibration remains "
            "Invalid."
        ),
        typical_writeback=(
            "No writeback. The summary reports 'best_value', the best gain for the "
            "repeated pulse; update the pulse module yourself."
        ),
        recommended=_SCAN_RECOMMENDED,
    )


class ZigZagScanFreqAdapter(_ZigZagScanBase):
    form = FREQ_FORM
    experiment_tag: ClassVar[str | None] = ZigZagScanExp.persistence_tag()
    sweep_key: ClassVar[str] = "freq"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Zig-zag frequency scan: runs the zig-zag repetition sequence at each drive "
            "frequency of the repeated pulse, and picks the frequency where the signal "
            "stays flattest across repetitions. Runs on real hardware. The simulator "
            "measures drive phase from the start of each shot, while hardware uses "
            "absolute time."
        ),
        expects_md=(
            "The gain scan runs 0.8 to 1.2 * qubit.module.x180.gain; for X90 repetition "
            "adjust the window to x90. The frequency scan runs qubit.freq ± 2 MHz. Both "
            "use 101 points and maximum repetitions 6. Relax delay is 5 * qubit.t1 us. "
            "Missing calibration leaves the affected fields empty and Invalid; no "
            "numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). Drive slots "
            "use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. "
            "Unset or dangling slots use the named template, not a legacy blank. Key "
            "choices remain available. Inline templates read qubit.freq, "
            "qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and "
            "resonator.ext.time_of_fly + 0.05 us; missing required calibration remains "
            "Invalid."
        ),
        typical_writeback=(
            "No writeback. The summary reports 'best_value', the best drive frequency "
            "(MHz); update 'qubit.freq' or the pulse module yourself."
        ),
        recommended=_SCAN_RECOMMENDED,
    )
