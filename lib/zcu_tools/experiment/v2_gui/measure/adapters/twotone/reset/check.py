from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

RabiCheckRunResult: TypeAlias = RunRecord[RabiCheckCfg, RabiCheckResult]


@dataclass
class RabiCheckAnalyzeResult(AnalyzeResultBase):
    frequency_cycles_per_gain: float
    before_amplitude: float
    after_amplitude: float
    relative_contrast: float
    before_phase_deg: float | None
    after_phase_deg: float | None
    phase_difference_deg: float | None
    reset_residual_amplitude: float
    reset_offset: float
    after_second_harmonic_amplitude: float
    before_residual_rms: float
    reset_residual_rms: float
    after_residual_rms: float


class RabiCheckAdapter(
    BaseAdapter[
        RabiCheckCfg, RabiCheckRunResult, RabiCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    """Rabi-amplitude check for any reset type (single-tone / two-pulse / bath).

    Sweeps the initialisation pulse gain and records three signal branches in
    parallel (without tested_reset / with tested_reset / with tested_reset + rabi_pulse),
    fitting relative Rabi contrast and residual input dependence. Averaged IQ
    diagnostics do not determine reset fidelity or a unique reset channel.
    """

    exp_cls = RabiCheckExp
    ExpCfg_cls: ClassVar[Any] = RabiCheckCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, analysis=AnalysisMode.FIT, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Reset Rabi check: sweeps the initialisation (rabi) pulse gain "
            "and acquires three branches simultaneously — without the tested reset, "
            "with the tested reset, and with the tested reset followed by another "
            "rabi pulse at the same swept gain. Fits the before-reset frequency "
            "and uses it for all branches, adding a second harmonic after reset. "
            "Reports half peak-to-peak amplitudes, relative contrast, phases, "
            "reset-only residual oscillation and residual RMS on one shared IQ axis. "
            "These are averaged-readout diagnostics, not reset fidelity. "
            "Runs on real hardware."
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
            "No writeback. Compare relative contrast, reset-only residual "
            "oscillation and fit residuals. Second-harmonic amplitude does not "
            "uniquely identify coherence or reset error; a flat reset-only trace "
            "does not establish ground-state preparation."
        ),
        recommended=(
            "Use at least six finite points per branch, preferably 51 points "
            "spanning a full Rabi period with enough density to resolve 2f. "
            "Frequency is in cycles/gain, phases in degrees at gain zero; "
            "phase is unavailable when the fundamental is unresolved above "
            "residual noise. Partial sweeps spanning less than a period may "
            "give poorly constrained fits. relax_delay should be long enough for thermal "
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

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> RabiCheckRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = RabiCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[RabiCheckRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> RabiCheckAnalyzeResult:
        fit = RabiCheckExp().analyze(req.run_result, None, plots=plots)
        return RabiCheckAnalyzeResult(
            frequency_cycles_per_gain=fit.frequency,
            before_amplitude=fit.before.amplitude,
            after_amplitude=fit.after.amplitude,
            relative_contrast=fit.contrast_ratio,
            before_phase_deg=fit.before.phase_deg,
            after_phase_deg=fit.after.phase_deg,
            phase_difference_deg=fit.phase_difference_deg,
            reset_residual_amplitude=fit.reset.amplitude,
            reset_offset=fit.reset.offset,
            after_second_harmonic_amplitude=fit.after.second_harmonic_amplitude,
            before_residual_rms=fit.before.residual_rms,
            reset_residual_rms=fit.reset.residual_rms,
            after_residual_rms=fit.after.residual_rms,
        )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_reset_check_{time.strftime('%m%d')}"
