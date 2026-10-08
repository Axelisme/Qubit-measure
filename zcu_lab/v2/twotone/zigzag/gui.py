"""GUI adapter for the plain ZigZag repetition experiment."""

from __future__ import annotations

import time
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from zcu_lab.v2._support.measure import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    scaled_md,
)
from zcu_lab.v2._support.measure.zigzag import (
    EXPECTS_ML,
    add_gate_modules,
    add_repeat_fields,
)
from zcu_lab.v2.twotone.zigzag.core import ZigZagCfg, ZigZagExp, ZigZagResult

ZigZagRunResult: TypeAlias = RunRecord[ZigZagCfg, ZigZagResult]


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
        expects_ml=EXPECTS_ML,
        typical_writeback=(
            "No analysis and no writeback. Preparation coherence, contrast, "
            "and drive history can also change the trace; compare controls "
            "before changing gate calibration."
        ),
        recommended=(
            "Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse "
            "(applied in pairs) to check the pi/2 pulse. About 10 repetitions "
            "usually show a mis-calibration clearly. Use the zig-zag scan "
            "adapters to sweep the repeated pulse's gain or frequency. "
            "reset_phase_cycle alternates the final reset pi phase by 180 degrees "
            "between averaging sweeps (even reps, two-pulse/bath reset). It adds "
            "no RF pulse or programmed wait; verify population and phase witnesses."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        builder = add_gate_modules(MeasureCfgBuilder()).relax_delay(
            scaled_md("t1", factor=5.0, fallback_value=30.5)
        )
        return (
            add_repeat_fields(builder, n_times=10)
            .bool(
                "reset_phase_cycle",
                label="Cycle reset pi phase (0/180)",
                default=False,
                tooltip="Alternate the final reset pulse by 180 degrees between averaging sweeps; requires even reps. Does not correct population or repeated-gate errors.",
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> ZigZagRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = ZigZagExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_zigzag_{time.strftime('%m%d')}"
