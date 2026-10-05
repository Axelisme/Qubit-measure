from __future__ import annotations

import time
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg
from zcu_lab.v2.onetone.power_dep.core import PowerDepExp
from zcu_lab.v2.onetone.power_dep.core import PowerDepResult
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.schema_builder import MeasureCfgDefinition
from zcu_lab.v2._support.measure.schema_builder import ModuleInit
from zcu_lab.v2._support.measure.seeds import res_freq_range
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import SweepValue

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    exp_cls = PowerDepExp
    ExpCfg_cls = PowerDepCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "One-tone resonator power dependence: a 2D sweep of readout power "
            "(gain) versus readout frequency, mapping how the resonator "
            "response shifts with drive strength. Used to find the punch-out / "
            "high-power transition and to pick a good low-power readout gain. "
            "Runs on real hardware; requires a SoC connection."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, centring the frequency sweep and setting the readout / "
            "ADC frequency (~4000–8000 MHz); 'rf_w' — linewidth, setting the "
            "span as r_f ± 1.5*rf_w (~5–50 MHz; falls back to ±30 MHz when "
            "absent); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — "
            "cable time-of-flight for the trigger offset."
        ),
        expects_ml=(
            "Needs a pulse-readout module, and references a ModuleLibrary "
            "waveform named 'ro_waveform' when present (optional)."
        ),
        typical_writeback=(
            "No writeback — this adapter has no analysis step (the underlying "
            "experiment is measurement-only). It produces a 2D map for "
            "visual inspection only; read off the punch-out power and "
            "low-power frequency by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis. Typical sweep: gain ~0.001 to 0.5 over ~101 points "
            "(low to high power), frequency r_f ± a couple of linewidths over "
            "~201 points. 'earlystop_snr' enters the run config (0 disables): "
            "set it positive to stop a column early once a signal-to-noise "
            "target is reached, speeding up the scan. Narrow the gain range "
            "once you've located the transition."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                    "pulse_cfg.gain": 0.0,
                },
            )
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.001, stop=1.0, expts=101),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=201),
            )
            .float(
                "earlystop_snr",
                label="Early-stop SNR (0 disables)",
                default=0.0,
                decimals=3,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> PowerDepCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("earlystop_snr", None)
        cfg = super().build_exp_cfg(cfg_raw, req)
        cfg.earlystop_snr = self._earlystop_snr(raw_cfg)
        return cfg

    def _earlystop_snr(self, raw_cfg: dict[str, object]) -> float | None:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> OneTonePowerDepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
