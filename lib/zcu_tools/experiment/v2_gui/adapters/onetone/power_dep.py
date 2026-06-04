from __future__ import annotations

import time

from typing_extensions import ClassVar, Optional, TypeAlias

from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_readout_module_spec,
    make_readout_default,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)

OneTonePowerDepRunResult: TypeAlias = PowerDepResult


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    exp_cls = PowerDepExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, supports_analysis=False
    )

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        # No reset module — one-tone runs without a qubit reset
                        # (the ExpCfg defaults reset=None).
                        "readout": make_pulse_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "earlystop_snr": ScalarSpec(
                    label="Early-stop SNR (0 disables)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={
                        "gain": SweepSpec(label="Gain (a.u.)"),
                        "freq": SweepSpec(label="Freq (MHz)"),
                    },
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_readout_default(ctx),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(10),
                "relax_delay": DirectValue(10.0),
                "earlystop_snr": DirectValue(0.0),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=0.001, stop=0.5, expts=101),
                        "freq": proper_res_freq_range(ctx, 201),
                    }
                ),
            }
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> PowerDepCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("earlystop_snr", None)
        return req.ml.make_cfg(cfg_raw, PowerDepCfg)

    def _earlystop_snr(self, raw_cfg: dict[str, object]) -> Optional[float]:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(self, req: RunRequest, schema: CfgSchema) -> OneTonePowerDepRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        earlystop_snr = self._earlystop_snr(raw_cfg)
        return PowerDepExp().run(soc, soccfg, cfg, earlystop_snr=earlystop_snr)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
