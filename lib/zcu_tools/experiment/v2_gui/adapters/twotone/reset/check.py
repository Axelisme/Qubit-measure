from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    run_figure_only_analyze,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
)
from zcu_tools.gui.cfg import (
    SweepValue,
)


@dataclass
class RabiCheckAnalyzeResult(FigureOnlyAnalyzeResult):
    pass


class RabiCheckAdapter(
    BaseAdapter[RabiCheckCfg, RabiCheckResult, RabiCheckAnalyzeResult, NoAnalyzeParams]
):
    """Rabi-amplitude check for any reset type (single-tone / two-pulse / bath).

    Sweeps the initialisation pulse gain and records three signal branches in
    parallel (without tested_reset / with tested_reset / with tested_reset + rabi_pulse),
    letting the user judge reset efficacy by eye. No automated fit — the three
    branches are visually compared (D5 / ADR-0011).
    """

    exp_cls = RabiCheckExp
    ExpCfg_cls: ClassVar[Any] = RabiCheckCfg
    # FIT enables the Analyze pane; the result contains a figure but no fitted scalar.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, analysis=AnalysisMode.FIT, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Reset Rabi check: sweeps the initialisation (rabi) pulse gain "
            "and acquires three branches simultaneously — without the tested reset, "
            "with the tested reset, and with the tested reset followed by another "
            "rabi pulse at the same swept gain — to judge how well reset prepares the "
            "ground state. Runs on real hardware. No automated fit; read the "
            "three traces by eye."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'q_f' / 'qub_ch' — "
            "qubit frequency and channel, seeding rabi_pulse and tested_reset "
            "defaults; 'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' "
            "— resonator / readout defaults."
        ),
        expects_ml=(
            "Needs rabi_pulse (used before and after tested_reset), tested_reset "
            "(any reset shape: none / pulse / two-pulse / bath), and a readout "
            "module. Optionally "
            "references a calibrated upstream reset (disabled when absent)."
        ),
        typical_writeback=(
            "No writeback. Inspect the three labeled analysis traces manually "
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("rabi_pulse", role_id="pi_pulse", label="Rabi Pulse")
            .reset("tested_reset", role_id="reset", label="Tested Reset")
            .readout()
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .reps(1000)
            .rounds(10)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[RabiCheckResult, NoAnalyzeParams]
    ) -> RabiCheckAnalyzeResult:
        return run_figure_only_analyze(RabiCheckExp, RabiCheckAnalyzeResult, req)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_reset_check_{time.strftime('%m%d')}"
