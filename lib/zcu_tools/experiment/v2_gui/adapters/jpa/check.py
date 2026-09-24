"""jpa/check GUI adapter — JPA pump off/on resonator comparison check.

Owns the jpa/check cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_tools.experiment.v2.jpa``
(``CheckExp``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from zcu_tools.experiment.v2.jpa import CheckCfg, CheckExp
from zcu_tools.experiment.v2.jpa.jpa_check import CheckResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    res_freq_range,
    run_figure_only_analyze,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
)

from ._shared import cached_device_snapshot, lower_jpa_rf_output_dev

# Bring-up survey: ~101 readout-frequency points around the resonator. These
# are inspectable starting bounds, NOT safety certification — the operator must
# review device and sweep.
_JPA_CHECK_FREQ_EXPTS = 101


@dataclass
class JpaCheckAnalyzeResult(FigureOnlyAnalyzeResult):
    # The check is look-at-the-comparison: the domain analyze renders the
    # pump-off vs pump-on resonator traces and extracts no writeback-able
    # scalar. ``figure`` is inherited.
    pass


class JpaCheckAdapter(
    BaseAdapter[CheckCfg, CheckResult, JpaCheckAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = CheckExp
    ExpCfg_cls: ClassVar[Any] = CheckCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA pump off/on resonator check: sweeps the readout frequency "
            "with the JPA pump RF output off and again with it on, recording "
            "the one-tone resonator response in both states so you can "
            "compare the resonance by eye. Runs on real hardware and commands "
            "the selected JPA RF device (its output is toggled off, then on). "
            "WARNING: the run leaves the JPA pump output ON when it finishes "
            "— the pump is NOT turned off afterwards. Review the selected RF "
            "device and the sweep range before running — the seeded bounds "
            "are bring-up defaults, not certified safety limits."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' — resonator "
            "frequency, centring the readout frequency sweep; 'rf_w' — "
            "linewidth, setting the span as r_f ± 1.5*rf_w; 'res_ch' / "
            "'ro_ch' — drive / ADC channels; 'timeFly' — cable time-of-flight "
            "for the trigger offset."
        ),
        expects_ml=(
            "Needs a pulse-readout module (e.g. 'readout_rf'); references a "
            "ModuleLibrary waveform named 'ro_waveform' when present. "
            "Optionally references a reset module."
        ),
        typical_writeback=(
            "No writeback — the check is a visual diagnostic: compare the two "
            "traces and update parameters in another step."
        ),
        recommended=(
            "No analysis options — the Analyze tab shows the pump off/on "
            "comparison figure only. Typical sweep: readout frequency ~101 "
            "points spanning 1.5 linewidths around 'r_f'. Before every run, "
            "confirm the current JPA pump state and review the selected RF "
            "device: the run toggles its output off then on and leaves it ON."
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
            .relax_delay(0.5)
            .device(
                "jpa_rf_dev",
                label="JPA RF device",
                default="",
                required=False,
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=_JPA_CHECK_FREQ_EXPTS),
            )
            .reps(1000)
            .rounds(5)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> CheckCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = lower_jpa_rf_output_dev(cfg_raw, cached_device_snapshot())
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req
        # Pure preflight over cached/static data — never commands a live device.
        lower_jpa_rf_output_dev(raw_cfg, cached_device_snapshot())

    def analyze(
        self, req: AnalyzeRequest[CheckResult, NoAnalyzeParams]
    ) -> JpaCheckAnalyzeResult:
        return run_figure_only_analyze(CheckExp, JpaCheckAnalyzeResult, req)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_jpa_check_{time.strftime('%m%d')}"
