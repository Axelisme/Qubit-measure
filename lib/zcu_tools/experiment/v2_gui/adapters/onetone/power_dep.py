from __future__ import annotations

import time
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    RoleInit,
    build_exp_spec,
    make_pulse_readout_module_spec,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    FloatSpec,
    RunRequest,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict

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
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                # No reset module — one-tone runs without a qubit reset
                # (the ExpCfg defaults reset=None). The sweep axes own
                # readout freq + gain (set_param at run; "freq" writes
                # both pulse and ro freq), so lock them off the form.
                "readout": make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0)
                .lock_literal("pulse_cfg.gain", 0.0),
            },
            sweep={
                "gain": SweepSpec(label="Gain (a.u.)"),
                "freq": SweepSpec(label="Freq (MHz)"),
            },
            extra={
                "earlystop_snr": FloatSpec(
                    label="Early-stop SNR (0 disables)", decimals=3
                )
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=1.0, earlystop_snr=0.0)
            .role("modules.readout", "readout", RoleInit.INLINE)
            .sweep("sweep.gain", SweepValue(start=0.001, stop=1.0, expts=101))
            .sweep("sweep.freq", proper_res_freq_range(ctx, 201))
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
