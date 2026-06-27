from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import (
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_res_freq_range,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    IntSpec,
    MetaDictWriteback,
    RunRequest,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)

RoOptAutoRunResult: TypeAlias = AutoOptResult


@dataclass
class RoOptAutoAnalyzeParams:
    # analyze() takes no tuning knobs — it reads off the best of the optimized
    # points. No form fields.
    pass


@dataclass
class RoOptAutoAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float
    best_length: float
    figure: Figure


class RoOptAutoAdapter(
    BaseAdapter[
        AutoOptCfg,
        RoOptAutoRunResult,
        RoOptAutoAnalyzeResult,
        RoOptAutoAnalyzeParams,
    ]
):
    exp_cls = AutoOptExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Automatic readout optimization: a Bayesian optimizer (skopt) "
            "searches readout frequency, gain and length jointly to maximize "
            "the g/e signal-to-noise ratio (SNR), with the qubit toggled "
            "between g and e by a pi pulse. Runs on real hardware. Use it to "
            "find a good readout in one shot instead of tuning freq / power / "
            "length one axis at a time."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
            "resonator / chosen readout frequency centring the freq search "
            "(~4000–8000 MHz); 'rf_w' — linewidth, scaling the freq search "
            "range (~5–50 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; "
            "'timeFly' — trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit "
            "frequency / channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a readout module (e.g. 'readout_rf'); "
            "references a ModuleLibrary waveform 'ro_waveform' when present. "
            "Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the optimizer's best readout frequency, gain and length "
            "into MetaDict 'best_ro_freq' (MHz), 'best_ro_gain' (a.u.) and "
            "'best_ro_length' (us). No ModuleLibrary writeback — combine them "
            "into a 'readout_dpm' module afterwards (the 'readout_dpm' role)."
        ),
        recommended=(
            "The three sweep axes define the optimizer's SEARCH BOUNDS (min / "
            "max), not a grid — keep them reasonably tight (e.g. freq within a "
            "fraction of a linewidth, gain and length in a sensible band) so "
            "the search converges. 'Optimizer points' (~1000) sets how many "
            "evaluations to spend; raise it for wider bounds, lower it for a "
            "quick search. No analysis knobs — it reports the best evaluated "
            "point."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                # The optimizer owns readout freq + gain (set_param at
                # run; "freq" writes both pulse and ro freq), so lock
                # them off the form. Length is swept into the pulse
                # waveform, not a top-level field — left editable.
                "readout": make_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0)
                .lock_literal("pulse_cfg.gain", 0.0),
            },
            sweep_label="Search bounds (min–max)",
            sweep={
                "freq": SweepSpec(label="Readout freq (MHz)"),
                "gain": SweepSpec(label="Readout gain (a.u.)"),
                "length": SweepSpec(label="Readout length (us)"),
            },
            extra={"num_points": IntSpec(label="Optimizer points")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=10, relax_delay=1.0, num_points=1000)
            .role("modules.qub_pulse", "qub_probe", Init.INLINE)
            .role("modules.readout", "readout")
            .role("modules.reset", "reset", Init.DISABLED)
            .set_sweep("sweep.freq", proper_res_freq_range(ctx, 51, span_factor=0.2))
            .sweep("sweep.gain", 0.1, 0.25, 51)
            .sweep("sweep.length", 5.0, 10.0, 51)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> AutoOptCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("num_points", None)
        return req.ml.make_cfg(cfg_raw, AutoOptCfg)

    def _num_points(self, raw_cfg: dict[str, object]) -> int:
        value = raw_cfg.get("num_points")
        if not isinstance(value, int):
            raise ValueError("num_points must be an integer")
        if value <= 0:
            raise ValueError("num_points must be positive")
        return value

    def run(self, req: RunRequest, schema: CfgSchema) -> RoOptAutoRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        num_points = self._num_points(raw_cfg)
        return AutoOptExp().run(soc, soccfg, cfg, num_points=num_points)

    def get_analyze_params(
        self, result: RoOptAutoRunResult, ctx: ExpContext
    ) -> RoOptAutoAnalyzeParams:
        return RoOptAutoAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[RoOptAutoRunResult, RoOptAutoAnalyzeParams]
    ) -> RoOptAutoAnalyzeResult:
        best_freq, best_gain, best_length, fig = AutoOptExp().analyze(req.run_result)
        return RoOptAutoAnalyzeResult(
            best_freq=best_freq,
            best_gain=best_gain,
            best_length=best_length,
            figure=fig,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptAutoRunResult, RoOptAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            MetaDictWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
            MetaDictWriteback(
                target_name="best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_auto_{time.strftime('%m%d')}"
