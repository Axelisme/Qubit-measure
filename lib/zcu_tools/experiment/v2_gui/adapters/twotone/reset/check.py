from __future__ import annotations

import time
from typing import Any, ClassVar

from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    SweepSpec,
)


class RabiCheckAdapter(BaseAdapter[RabiCheckCfg, RabiCheckResult]):
    """Rabi-amplitude check for any reset type (single-tone / two-pulse / bath).

    Sweeps the initialisation pulse gain and records three signal branches in
    parallel (without reset / with tested_reset / with tested_reset + pi_pulse),
    letting the user judge reset efficacy by eye. No automated fit — the three
    branches are visually compared (D5 / ADR-0011).
    """

    exp_cls = RabiCheckExp
    ExpCfg_cls: ClassVar[Any] = RabiCheckCfg
    # No automated fit: the three-branch plot is read by eye (D5).
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, analysis=AnalysisMode.NONE
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Reset Rabi check: sweeps the initialisation (rabi) pulse gain "
            "and acquires three branches simultaneously — without any reset, "
            "with the tested_reset, and with tested_reset followed by a pi "
            "pulse — to let the user judge how well the reset prepares the "
            "ground state. Runs on real hardware. No automated fit; read the "
            "three traces by eye."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'q_f' / 'qub_ch' — "
            "qubit frequency and channel, seeding rabi_pulse / pi_pulse / "
            "tested_reset defaults; 'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' "
            "— resonator / readout defaults."
        ),
        expects_ml=(
            "Needs rabi_pulse (init pulse — typically pi_amp), tested_reset "
            "(any reset shape: none / pulse / two-pulse / bath), pi_pulse "
            "(calibrated pi pulse), and a readout module. Optionally "
            "references a calibrated upstream reset (disabled when absent)."
        ),
        typical_writeback=(
            "No analysis and no writeback. Inspect the three traces manually "
            "to confirm the reset prepares the ground state; proceed to the "
            "next calibration step once satisfied."
        ),
        recommended=(
            "A gain sweep of 51 points from 0.0 to 1.0 captures the full "
            "pi-pulse rotation. relax_delay should be long enough for thermal "
            "equilibration (notebook: 5 × T1 for bath reset). No reset is "
            "optional — omit it if no upstream reset is calibrated yet."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                # Optional upstream reset applied before the rabi init pulse.
                "reset": make_reset_module_spec(optional=True),
                # Init pulse whose gain is swept; typically a pi pulse (pi_amp).
                "rabi_pulse": make_pulse_module_spec(label="Rabi Pulse"),
                # The reset under test — all four shapes so the user can select
                # single-tone / two-pulse / bath without opening a different tab.
                "tested_reset": make_reset_module_spec(label="Tested Reset"),
                # Pi pulse applied after tested_reset in the third branch.
                "pi_pulse": make_pulse_module_spec(label="Pi Pulse"),
                "readout": make_readout_module_spec(),
            },
            sweep={"gain": SweepSpec(label="Gain (a.u.)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=10, relax_delay=1.0)
            # Library pi_amp is the canonical init pulse for the rabi check
            # (notebook: rabi_pulse = "pi_amp").
            .role("modules.rabi_pulse", "pi_pulse")
            .role("modules.readout", "readout")
            # tested_reset defaults to pulse_reset; the user switches shape in
            # the form when checking a two-pulse or bath reset.
            .role("modules.tested_reset", "reset")
            # pi_pulse for the third branch (same library pi_amp).
            .role("modules.pi_pulse", "pi_pulse")
            # optional upstream reset — None when no library entry (ADR-0010).
            .role("modules.reset", "reset", Init.DISABLED)
            .sweep("sweep.gain", 0.0, 1.0, 51)
            .build()
        )

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_reset_check_{time.strftime('%m%d')}"
