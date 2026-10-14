from __future__ import annotations

import time
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg, PowerDepExp, PowerDepResult
from zcu_lab.v2.onetone.power_dep.form import make_form

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    experiment_tag: ClassVar[str | None] = PowerDepExp.persistence_tag()
    exp_cls = PowerDepExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

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
            "Reads fixed tab parameters: resonator.freq and resonator.kappa "
            "seed the frequency sweep; resonator.wiring.ch/ro_ch seed the "
            "generator/ADC channels. Captured resonator.ext.default_probe_length "
            "minus 0.1 us seeds the ADC window, and resonator.ext.time_of_fly "
            "plus 0.05 us seeds trigger timing. Missing or null calibration "
            "and unbound aliases leave required fields Invalid without numeric "
            "fallback; source failures propagate."
        ),
        expects_ml=(
            "Starts with a fresh inline pulse readout and const waveform. "
            "It does not adopt a library module or named waveform."
        ),
        typical_writeback=(
            "No writeback — this adapter has no analysis step (the underlying "
            "experiment is measurement-only). It produces a 2D map for "
            "visual inspection only; read off the punch-out power and "
            "low-power frequency by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis. Typical sweep: gain ~0.001 to 0.5 over ~101 points "
            "(low to high power), frequency resonator.freq ± linewidths over "
            "~201 points. 'earlystop_snr' starts disabled (None or nonpositive): "
            "set it positive to stop a column early once a signal-to-noise "
            "target is reached, speeding up the scan. Narrow the gain range "
            "once you've located the transition."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: PowerDepCfg,
        *,
        context: RunContext,
    ) -> OneTonePowerDepRunResult:
        """Acquire a power/frequency map from accepted cfg values.

        cfg is this adapter's validated Form output. Nonpositive earlystop_snr
        disables stopping on a detached copy; context supplies live SoC/device
        handles, cancellation and plots. req is unused.
        Return this run's typed cfg and result. Acquisition errors propagate.
        """
        del req
        cfg = cfg.with_updates(
            earlystop_snr=cfg.earlystop_snr
            if cfg.earlystop_snr is not None and cfg.earlystop_snr > 0
            else None
        )
        result = PowerDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
