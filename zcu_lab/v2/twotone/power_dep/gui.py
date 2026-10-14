from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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

from zcu_lab.v2.twotone.power_dep.core import PowerCfg, PowerExp, PowerResult

from .form import FORM

PowerDepRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


class PowerDepAdapter(BaseAdapter[PowerCfg, PowerDepRunResult]):
    form = FORM
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Two-tone qubit power dependence: a 2D scan sweeping the qubit-drive gain "
            "(outer) against the drive frequency (inner), reading out the resonator at "
            "each point, to map how the qubit line shifts and broadens with drive power "
            "(AC-Stark shift, power broadening, multi-photon lines). Runs on real "
            "hardware. No automated fit — read off the 2D map by eye."
        ),
        expects_md=(
            "Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, "
            "resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the "
            "fixed capture. Missing required calibration remains Invalid; no numeric "
            "fallback. Frequency sweep is qubit.freq ± 1.5 * qubit.kappa; gain runs "
            "0.001 to 1.0."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a fresh readout template when "
            "unset or dangling. Optional qubit.module.reset stays disabled without a "
            "source. Library references remain selectable; sweeps overwrite their "
            "controlled fields only at Run. Qubit probe starts inline at template "
            "gain/frequency; both are overwritten by the sweeps."
        ),
        typical_writeback=(
            "No analysis and no writeback (the underlying experiment raises "
            "NotImplementedError). Inspect the 2D map manually; pick the low-power "
            "qubit frequency to feed back into 'qubit.freq' yourself."
        ),
        recommended=(
            "No fit options. Default sweep: gain ~0.001→1.0 over ~101 points, frequency "
            "~201 points spanning 1.5 linewidths around 'qubit.freq'. Start the gain "
            "sweep low to find the unsaturated line, then extend the upper bound to "
            "watch the AC-Stark shift and power broadening; widen the frequency span if "
            "higher-power features fall outside the window."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> PowerDepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"
