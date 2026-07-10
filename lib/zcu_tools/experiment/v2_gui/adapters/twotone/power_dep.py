from __future__ import annotations

import time
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    RoleInit,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_qub_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ExpContext,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    SweepSpec,
    SweepValue,
)

PowerDepRunResult: TypeAlias = PowerResult


class PowerDepAdapter(BaseAdapter[PowerCfg, PowerDepRunResult]):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Two-tone qubit power dependence: a 2D scan sweeping the "
            "qubit-drive gain (outer) against the drive frequency (inner), "
            "reading out the resonator at each point, to map how the qubit "
            "line shifts and broadens with drive power (AC-Stark shift, power "
            "broadening, multi-photon lines). Runs on real hardware. No "
            "automated fit — read off the 2D map by eye."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'q_f' — qubit frequency, "
            "centring the inner sweep (~2000–6000 MHz); 'qf_w' — qubit "
            "linewidth, half-span 1.5*qf_w (~1–50 MHz); 'qub_ch' — qubit-drive "
            "channel; 'r_f' — resonator frequency for the readout tone "
            "(~4000–8000 MHz); 'res_ch' / 'ro_ch' — readout drive / ADC "
            "channels; 'timeFly' — readout trigger-offset cable delay (~0–1 "
            "us). Absent 'q_f'/'qf_w' → ±30 MHz around 4000 MHz."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module and a pulse-readout module; "
            "references a ModuleLibrary waveform named 'ro_waveform' when "
            "present. Optionally references a calibrated reset module "
            "(disabled when none exists)."
        ),
        typical_writeback=(
            "No analysis and no writeback (the underlying experiment raises "
            "NotImplementedError). Inspect the 2D map manually; pick the "
            "low-power qubit frequency to feed back into 'q_f' yourself."
        ),
        recommended=(
            "No fit options. Default sweep: gain ~0.001→0.5 over ~101 points, "
            "frequency ~201 points spanning 1.5 linewidths around 'q_f'. Start "
            "the gain sweep low to find the unsaturated line, then extend the "
            "upper bound to watch the AC-Stark shift and power broadening; "
            "widen the frequency span if higher-power features fall outside "
            "the window."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                # Both sweep axes own qubit-drive freq + gain (set_param
                # at run); lock so the form hides the silently-overwritten
                # fields.
                "qub_pulse": make_pulse_module_spec()
                .lock_literal("freq", 0.0)
                .lock_literal("gain", 0.0),
                "readout": make_readout_module_spec(),
            },
            sweep={
                "gain": SweepSpec(label="Gain (a.u.)"),
                "freq": SweepSpec(label="Freq (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=1.0)
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", RoleInit.DISABLED)
            .role("modules.qub_pulse", "qub_probe", RoleInit.INLINE)
            .role("modules.readout", "readout")
            .sweep("sweep.gain", SweepValue(start=0.001, stop=1.0, expts=101))
            .sweep("sweep.freq", proper_qub_freq_range(ctx, 201))
            .build()
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"
