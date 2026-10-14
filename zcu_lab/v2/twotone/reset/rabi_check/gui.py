from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
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
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.rabi_check.core import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckResult,
)
from zcu_lab.v2.twotone.reset.rabi_check.form import make_form

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

    experiment_tag: ClassVar[str | None] = RabiCheckExp.persistence_tag()

    exp_cls = RabiCheckExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[RabiCheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

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
            "Seeds 51 gain points from 0.0 to 1.0, with relax delay 1.0 us. Run "
            "overwrites the Rabi pulse gain. Open and Reset evaluate starting expressions "
            "once into direct input using captured sources. Inline qubit pulses read "
            "$qubit.freq and $qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "Rabi pulse uses $qubit.module.x180 or a fresh pi_pulse template. Tested "
            "reset uses $qubit.module.reset or a fresh pulse-reset template; library "
            "reset families remain selectable. $qubit.module.readout selects a captured "
            "library link, otherwise a fresh inline readout template with a const "
            "waveform. Its required parameter sources must still be present. Optional "
            "$qubit.module.reset has no fallback: an unset/dangling slot or unbound alias "
            "leaves it disabled and Valid."
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

    def run(
        self,
        req: RunRequest,
        cfg: RabiCheckCfg,
        *,
        context: RunContext,
    ) -> RabiCheckRunResult:
        del req
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
