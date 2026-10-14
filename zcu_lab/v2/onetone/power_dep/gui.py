from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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

from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg, PowerDepExp, PowerDepResult
from zcu_lab.v2.onetone.power_dep.form import FORM

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    experiment_tag: ClassVar[str | None] = PowerDepExp.persistence_tag()
    exp_cls = PowerDepExp
    ExpCfg_cls = PowerDepCfg
    form = FORM
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

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> PowerDepCfg:
        """Assemble accepted cfg values with req's captured device snapshot.

        raw_cfg is the resolved experiment tree from cfg acceptance. Positive
        numeric earlystop_snr enables stopping; nonpositive values disable it.
        Return the typed cfg without hardware I/O; assembly errors propagate.
        """
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("earlystop_snr", None)
        cfg = super().build_exp_cfg(cfg_raw, req)
        cfg.earlystop_snr = self._earlystop_snr(raw_cfg)
        return cfg

    def _earlystop_snr(self, raw_cfg: Mapping[str, ExperimentCfgValue]) -> float | None:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> OneTonePowerDepRunResult:
        """Acquire a power/frequency map from accepted cfg values.

        req supplies captured device settings for assembly; context supplies
        live SoC/device handles, cancellation and plots for acquisition.
        Return this run's typed cfg and result. Acquisition errors propagate.
        """
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
