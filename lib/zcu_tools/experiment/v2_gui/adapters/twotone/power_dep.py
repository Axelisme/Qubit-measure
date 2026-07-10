from __future__ import annotations

import time
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    qub_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ExpContext,
)
from zcu_tools.gui.cfg import (
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                locked={"freq": 0.0, "gain": 0.0},
            )
            .readout()
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.001, stop=1.0, expts=101),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=qub_freq_range(expts=201),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"
