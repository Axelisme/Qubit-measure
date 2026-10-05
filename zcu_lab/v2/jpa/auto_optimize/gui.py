"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from matplotlib.figure import Figure
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.jpa_shared import (
    lower_jpa_flux_dev,
    lower_jpa_rf_dev,
    lower_jpa_rf_power_dev,
)
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import custom
from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
)
from zcu_lab.v2.jpa.flux.gui import jpa_flux_sweep_seed
from zcu_lab.v2.jpa.freq.gui import jpa_freq_sweep_seed
from zcu_lab.v2.jpa.power.gui import jpa_power_sweep_seed

JpaAutoRunResult = RunRecord[JPAOptCfg, JPAOptimizeResult]

# Bring-up iteration budget (contract: GUI default may use 1001).
_JPA_AUTO_NUM_POINTS_DEFAULT = 1001


def _lower_jpa_auto_devs(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
) -> dict[str, dict[str, str]]:
    """Lower both JPA selectors for the joint optimization.

    The RF device must support both the frequency and the power knob (the
    optimizer commands both on the same device); the flux device must support
    the flux knob. All checks run over the cached snapshot only — never a
    live device. A device cannot carry both roles at once: two different JPA
    roles on the same device would collapse the labeled patches, so a
    repeated selection fast-fails.
    """
    rf_patch = lower_jpa_rf_dev(raw_cfg, device_snapshot)
    # The same RF device must also carry the power knob.
    lower_jpa_rf_power_dev(raw_cfg, device_snapshot)
    flux_patch = lower_jpa_flux_dev(raw_cfg, device_snapshot)
    (rf_name,) = rf_patch
    (flux_name,) = flux_patch
    if rf_name == flux_name:
        raise ValueError(
            "JPA auto_optimize requires distinct RF and flux devices; "
            f"{rf_name!r} is selected for both 'jpa_rf_dev' and 'jpa_flux_dev'"
        )
    return {**rf_patch, **flux_patch}


def _relabel_auto_figure(fig: Figure) -> None:
    """Neutral 'JPA flux device value' wording on the GUI review figure.

    The core auto-optimize analysis figure labels its flux subplot 'JPA Flux
    value (a.u.)'; the GUI review figure must use the neutral vocabulary of
    the selected flux device without claiming a physical-unit migration. Only
    the flux axis is relabelled — the frequency and power axes keep their own
    quantities.
    """
    for ax in fig.axes:
        if "Flux" in ax.get_xlabel():
            ax.set_xlabel("JPA flux device value")


@dataclass
class JpaAutoAnalyzeResult(AnalyzeResultBase):
    best_flux: float
    best_freq: float
    best_power: float


class JpaAutoOptimizeAdapter(
    BaseAdapter[JPAOptCfg, JpaAutoRunResult, JpaAutoAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = AutoOptimizeExp
    ExpCfg_cls: ClassVar[Any] = JPAOptCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "JPA joint optimization: with the qubit prepared in g and e (a pi "
            "pulse toggles it), a multi-phase optimizer searches the JPA flux "
            "device value, pump frequency and pump power jointly to maximize "
            "the g/e signal difference. Runs on real hardware and commands "
            "the selected RF and flux devices at every evaluated point, "
            "leaving them at the last evaluated point. WARNING: review the "
            "selected devices and the search bounds before running — the "
            "seeded bounds are bring-up defaults, not certified safety "
            "limits."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'best_jpa_flux' — a "
            "previously accepted JPA flux device value, preferred as the flux "
            "search centre; 'best_jpa_freq' — a previously accepted pump "
            "frequency, preferred as the frequency search centre; "
            "'best_jpa_power' — a previously accepted pump power, preferred "
            "as the power search centre; 'r_f' — resonator frequency, "
            "seeding the pump search near 2*r_f; 'res_ch' / 'ro_ch' — drive "
            "/ ADC channels; 'timeFly' — cable time-of-flight for the trigger "
            "offset; 'q_f' / 'qub_ch' — qubit frequency / drive channel for "
            "the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform named "
            "'ro_waveform' when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the signal-maximizing flux device value, pump frequency "
            "and pump power into three separate MetaDict drafts — "
            "'best_jpa_flux', 'best_jpa_freq' (MHz) and 'best_jpa_power' "
            "(dBm) — so you can accept, cancel or retarget each one "
            "independently. Nothing is written without your acceptance, "
            "writeback never commands a device, and it never touches "
            "'cur_jpa_A'."
        ),
        recommended=(
            "For each sweep, start/stop are SEARCH BOUNDS and expts is a "
            "RELATIVE RESOLUTION HINT — expts is not a Cartesian or exact "
            "sample count. The optimizer compares all three hints when "
            "allocating its phase-1 flux slices and points per slice. "
            "'Global optimizer points' (num_points, at least 4, default "
            "1001) remains the total evaluation budget: raise it for wider "
            "bounds, lower it for a quick search. Analysis reports the best "
            "evaluated point with no tuning knobs."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        sweep_label = "Search bounds + relative resolution hints"
        sweep_tooltip = (
            "Start/stop set this axis's search bounds. Expts is a relative "
            "resolution hint compared with the other axes when allocating "
            "phase-1 flux slices; it is not a Cartesian or exact sample count."
        )
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .readout()
            .relax_delay(30.5)
            .device(
                "jpa_rf_dev",
                label="JPA RF device",
                default="",
                required=False,
            )
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
                    description="jpa flux device value search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .sweep(
                "jpa_freq",
                label="JPA pump freq (MHz)",
                default=custom(
                    jpa_freq_sweep_seed,
                    description="jpa pump frequency search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .sweep(
                "jpa_power",
                label="JPA pump power (dBm)",
                default=custom(
                    jpa_power_sweep_seed,
                    description="jpa pump power search bounds",
                ),
                section_label=sweep_label,
                tooltip=sweep_tooltip,
            )
            .int(
                "num_points",
                label="Global optimizer points",
                default=_JPA_AUTO_NUM_POINTS_DEFAULT,
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(1000)
            .rounds(1)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> JPAOptCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw["dev"] = _lower_jpa_auto_devs(cfg_raw, req.device_snapshot)
        return super().build_exp_cfg(cfg_raw, req)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        # Pure preflight over the detached request snapshot.
        self.build_exp_cfg(raw_cfg, req)

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> JpaAutoRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, AutoOptimizeExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaAutoRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaAutoAnalyzeResult:
        analysis = AutoOptimizeExp().analyze(req.run_result, None, plots=plots)
        _relabel_auto_figure(plots["fit"])
        return JpaAutoAnalyzeResult(
            best_flux=analysis.best_flux,
            best_freq=analysis.best_freq,
            best_power=analysis.best_power,
        )

    def get_writeback_items(
        self, req: WritebackRequest[JpaAutoRunResult, JpaAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="best_jpa_flux",
                description="Best JPA flux device value",
                proposed_value=result.best_flux,
            ),
            MetaDictWriteback(
                target_name="best_jpa_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            MetaDictWriteback(
                target_name="best_jpa_power",
                description="Best JPA pump power (dBm)",
                proposed_value=result.best_power,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_auto_{time.strftime('%m%d')}"
