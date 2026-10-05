"""Zig-zag pulse-error amplification adapters.

``ZigZagAdapter`` runs the plain repetition count scan and has no analysis.
``ZigZagScanExp`` sweeps exactly one of gain or freq on the repeated pulse; the
two scan adapters each expose one sweep so the lowered cfg always satisfies the
domain's "one sweep set" requirement, like the single-shot T1-tone sweep pair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.zigzag import ZigZagCfg, ZigZagExp, ZigZagResult
from zcu_tools.experiment.v2.twotone.zigzag_sweep import (
    ZigZagScanAnalyzeOptions,
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    Seed,
    SweepDefault,
    custom,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support.ctx_helpers import (
    md_get_float,
    md_has_key,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

ZigZagRunResult: TypeAlias = RunRecord[ZigZagCfg, ZigZagResult]
ZigZagScanRunResult: TypeAlias = RunRecord[ZigZagScanCfg, ZigZagScanResult]

_REPEAT_ON_CHOICES = ("X180_pulse", "X90_pulse")
_FREQ_HALF_SPAN = 2.0  # MHz around q_f for the frequency scan default

_EXPECTS_ML = (
    "Needs an X90 pulse module (prefers the calibrated library pi/2 pulse "
    "'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / "
    "'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' "
    "/ 'readout' / 'res_readout', else a blank pulse-readout). Optional reset "
    "references a library reset when present, else stays disabled."
)


def _modules(builder: MeasureCfgBuilder) -> MeasureCfgBuilder:
    return (
        builder.reset(optional=True)
        .pulse("X90_pulse", role_id="pi2_pulse", label="X90 Pulse")
        .pulse("X180_pulse", role_id="pi_pulse", label="X180 Pulse")
        .readout()
    )


def _repeat_fields(builder: MeasureCfgBuilder, *, n_times: int) -> MeasureCfgBuilder:
    return builder.int(
        "n_times",
        label="Max repetitions",
        default=n_times,
        tooltip="Repetition counts run from 0 to this value.",
    ).choice(
        "repeat_on",
        label="Repeat pulse",
        choices=_REPEAT_ON_CHOICES,
        default="X180_pulse",
        tooltip="X90_pulse repeats in pairs, so each count adds one pi rotation.",
    )


class ZigZagAdapter(BaseAdapter[ZigZagCfg, ZigZagRunResult]):
    exp_cls = ZigZagExp
    ExpCfg_cls: ClassVar[Any] = ZigZagCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Zig-zag: applies an X90 pulse, then repeats the chosen pulse "
            "0..n_times times before readout, amplifying small pulse-amplitude "
            "errors into a growing zig-zag of the signal versus repetition "
            "count. Runs on real hardware. Run after amplitude Rabi has "
            "calibrated the pi and pi/2 pulses."
        ),
        expects_md=(
            "Reads 't1' (us) to seed relax_delay as 5*t1 (fallback ~30 us). "
            "Pulse modules pull 'q_f' (~2000–6000 MHz) and 'qub_ch'; readout "
            "pulls 'r_f', 'res_ch' / 'ro_ch' and 'timeFly'."
        ),
        expects_ml=_EXPECTS_ML,
        typical_writeback=(
            "No analysis and no writeback. A flat trace means the repeated "
            "pulse is calibrated; a zig-zag or drift means it is off."
        ),
        recommended=(
            "Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse "
            "(applied in pairs) to check the pi/2 pulse. About 10 repetitions "
            "usually show a mis-calibration clearly. Use the zig-zag scan "
            "adapters to sweep the repeated pulse's gain or frequency."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        builder = _modules(MeasureCfgBuilder()).relax_delay(
            scaled_md("t1", factor=5.0, fallback_value=30.5)
        )
        return _repeat_fields(builder, n_times=10).reps(1000).rounds(100).build()

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> ZigZagRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ZigZagExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_zigzag_{time.strftime('%m%d')}"


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
            _modules(MeasureCfgBuilder())
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.5))
            .sweep(cls.sweep_key, label=cls.sweep_label, default=cls.sweep_default())
        )
        return _repeat_fields(builder, n_times=6).reps(100).rounds(100).build()

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
        expects_ml=_EXPECTS_ML,
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
        expects_ml=_EXPECTS_ML,
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
