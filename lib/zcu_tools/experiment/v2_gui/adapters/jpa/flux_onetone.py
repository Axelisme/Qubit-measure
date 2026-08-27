"""jpa/flux_onetone GUI adapter — JPA flux × readout-frequency 2D survey.

Owns the jpa/flux_onetone cfg definition, run/no-analysis policy and the
operator guide. The core experiment lives in
``zcu_tools.experiment.v2.jpa`` (``OneToneFluxExp``).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from zcu_tools.experiment.v2.jpa import OneToneFluxCfg, OneToneFluxExp
from zcu_tools.experiment.v2.jpa.jpa_flux_onetone import OneToneFluxResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    res_freq_range,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ExpContext,
    RunRequest,
)

from ._shared import cached_device_snapshot, lower_jpa_flux_dev
from .flux import jpa_flux_sweep_seed

# Bring-up survey: ~101 readout-frequency points around the resonator; the
# flux sweep reuses the shared JPA flux survey seed (101 points). These are
# inspectable starting bounds, NOT safety certification — the operator must
# review device and sweeps.
_JPA_FLUX_ONETONE_FREQ_EXPTS = 101


class JpaFluxOneToneAdapter(BaseAdapter[OneToneFluxCfg, OneToneFluxResult]):
    exp_cls = OneToneFluxExp
    ExpCfg_cls: ClassVar[Any] = OneToneFluxCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA flux × readout-frequency 2D survey: sweeps the JPA flux "
            "device value (outer axis) against the readout frequency (inner "
            "axis) and records the one-tone resonator response, so you can "
            "see by eye how the JPA response changes with flux and frequency. "
            "There is no automatic optimum analysis — judge the 2D map "
            "directly. Runs on real hardware and commands the selected JPA "
            "flux device. WARNING: review the selected flux device and both "
            "sweep ranges before running — the seeded bounds are bring-up "
            "defaults, not certified safety limits. The JPA pump is NOT "
            "controlled by this experiment: if the pump must be on for the "
            "measurement, confirm its state yourself (in the Devices "
            "interface) before running."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'best_jpa_flux' — a "
            "previously accepted JPA flux device value, preferred as the flux "
            "sweep centre; 'r_f' — resonator frequency, centring the readout "
            "frequency sweep; 'rf_w' — linewidth, setting the span as "
            "r_f ± 1.5*rf_w; 'res_ch' / 'ro_ch' — drive / ADC channels; "
            "'timeFly' — cable time-of-flight for the trigger offset."
        ),
        expects_ml=(
            "Needs a pulse-readout module (e.g. 'readout_rf'); references a "
            "ModuleLibrary waveform named 'ro_waveform' when present. "
            "Optionally references a reset module."
        ),
        typical_writeback=(
            "No writeback — this adapter has no analysis step (the "
            "underlying experiment has no optimum analysis yet). It produces "
            "a 2D map for visual inspection only; read off useful flux / "
            "frequency settings by eye and update parameters in another step."
        ),
        recommended=(
            "No analysis options. Typical sweep: JPA flux device value ~101 "
            "points around the seeded centre (best_jpa_flux when present), "
            "readout frequency ~101 points spanning 1.5 linewidths around "
            "'r_f'. Confirm the JPA pump state yourself before every run. "
            "Survey the map wide first, then narrow the ranges around the "
            "response you want to inspect."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .readout(
                pulse_only=True,
                init=ModuleInit.INLINE,
                locked={
                    "pulse_cfg.freq": 0.0,
                    "ro_cfg.ro_freq": 0.0,
                },
            )
            .relax_delay(0.1)
            .device(
                "jpa_flux_dev",
                label="JPA flux device",
                default="",
                required=False,
            )
            .sweep(
                "jpa_flux",
                label="JPA flux device value",
                default=custom(
                    jpa_flux_sweep_seed,
                    description="jpa flux device value range",
                ),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=_JPA_FLUX_ONETONE_FREQ_EXPTS),
            )
            .reps(100)
            .rounds(10)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: dict[str, object], req: RunRequest
    ) -> OneToneFluxCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_flux_dev(cfg_raw, cached_device_snapshot())
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req
        # Pure preflight over cached/static data — never commands a live device.
        lower_jpa_flux_dev(raw_cfg, cached_device_snapshot())

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_jpa_flux_onetone_{time.strftime('%m%d')}"
