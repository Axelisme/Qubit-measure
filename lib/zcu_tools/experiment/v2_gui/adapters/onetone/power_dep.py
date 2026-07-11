from __future__ import annotations

import time
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    res_freq_range,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ExpContext,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    SweepValue,
)

OneTonePowerDepRunResult: TypeAlias = PowerDepResult


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    exp_cls = PowerDepExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
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
            "experiment raises NotImplementedError). It produces a 2D map for "
            "visual inspection only; read off the punch-out power and "
            "low-power frequency by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis. Typical sweep: gain ~0.001 to 0.5 over ~101 points "
            "(low to high power), frequency r_f ± a couple of linewidths over "
            "~201 points. 'earlystop_snr' is a GUI-only control (0 disables): "
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
        return req.ml.make_cfg(cfg_raw, PowerDepCfg)

    def _earlystop_snr(self, raw_cfg: dict[str, object]) -> float | None:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(self, req: RunRequest, schema: CfgSchema) -> OneTonePowerDepRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        earlystop_snr = self._earlystop_snr(raw_cfg)
        return PowerDepExp().run(soc, soccfg, cfg, earlystop_snr=earlystop_snr)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
