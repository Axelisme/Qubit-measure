"""GUI adapters for gain and frequency scans of the ZigZag experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

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
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    Seed,
    SweepDefault,
    custom,
    scaled_md,
)
from zcu_lab.v2._support.measure.ctx_helpers import md_get_float, md_has_key
from zcu_lab.v2._support.measure.zigzag import (
    EXPECTS_ML,
    add_gate_modules,
    add_repeat_fields,
)
from zcu_lab.v2.twotone.zigzag_sweep.core import (
    ZigZagScanAnalyzeOptions,
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanResult,
)

ZigZagScanRunResult: TypeAlias = RunRecord[ZigZagScanCfg, ZigZagScanResult]

_FREQ_HALF_SPAN = 2.0  # MHz around q_f for the frequency scan default


@dataclass
class ZigZagScanAnalyzeParams:
    find_min: Annotated[float | None, ParamMeta(label="Search from")] = None
    find_max: Annotated[float | None, ParamMeta(label="Search to")] = None


@dataclass
class ZigZagScanAnalyzeResult(AnalyzeResultBase):
    best_value: float


def _qub_freq_window(ctx: SessionEnv, expts: int) -> SweepValue:
    if md_has_key(ctx, "q_f"):
        start: float | EvalValue = EvalValue(expr=f"q_f - {_FREQ_HALF_SPAN}")
        stop: float | EvalValue = EvalValue(expr=f"q_f + {_FREQ_HALF_SPAN}")
    else:
        center = md_get_float(ctx, "q_f", 5000.0)
        start, stop = center - _FREQ_HALF_SPAN, center + _FREQ_HALF_SPAN
    return SweepValue(start=start, stop=stop, expts=expts)


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
    sweep_label: ClassVar[str]

    @classmethod
    def sweep_default(cls) -> SweepDefault | Seed[SweepValue]:
        raise NotImplementedError

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        builder = (
            add_gate_modules(MeasureCfgBuilder())
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.5))
            .sweep(cls.sweep_key, label=cls.sweep_label, default=cls.sweep_default())
        )
        return add_repeat_fields(builder, n_times=6).reps(100).rounds(100).build()

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
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
    sweep_key: ClassVar[str] = "gain"
    sweep_label: ClassVar[str] = "Gain (a.u.)"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Zig-zag gain scan: runs the zig-zag repetition sequence at each "
            "gain of the repeated pulse, and picks the gain where the signal "
            "stays flattest across repetitions. Runs on real hardware."
        ),
        expects_md=(
            "Reads 'pi_gain' to seed the gain sweep as 0.8–1.2*pi_gain "
            "(fallback 0.4–0.6); with repeat on X90_pulse, set the sweep "
            "around 'pi2_gain' instead. 't1' seeds relax_delay as 5*t1."
        ),
        expects_ml=EXPECTS_ML,
        typical_writeback=(
            "No writeback. The summary reports 'best_value', the best gain "
            "for the repeated pulse; update the pulse module yourself."
        ),
        recommended=_SCAN_RECOMMENDED,
    )

    @classmethod
    def sweep_default(cls) -> SweepDefault:
        return SweepDefault(
            start=scaled_md("pi_gain", factor=0.8, fallback_value=0.4),
            stop=scaled_md("pi_gain", factor=1.2, fallback_value=0.6),
            expts=101,
        )


class ZigZagScanFreqAdapter(_ZigZagScanBase):
    sweep_key: ClassVar[str] = "freq"
    sweep_label: ClassVar[str] = "Frequency (MHz)"

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Zig-zag frequency scan: runs the zig-zag repetition sequence at "
            "each drive frequency of the repeated pulse, and picks the "
            "frequency where the signal stays flattest across repetitions. "
            "Runs on real hardware. The simulator measures drive phase from "
            "the start of each shot, while hardware uses absolute time."
        ),
        expects_md=(
            "Reads 'q_f' to seed the frequency sweep as q_f ± 2 MHz "
            "(fallback 4998–5002 MHz). 't1' seeds relax_delay as 5*t1."
        ),
        expects_ml=EXPECTS_ML,
        typical_writeback=(
            "No writeback. The summary reports 'best_value', the best drive "
            "frequency (MHz); update 'q_f' or the pulse module yourself."
        ),
        recommended=_SCAN_RECOMMENDED,
    )

    @classmethod
    def sweep_default(cls) -> Seed[SweepValue]:
        return custom(
            lambda ctx: _qub_freq_window(ctx, 101),
            description="qubit frequency window (101 points)",
        )
