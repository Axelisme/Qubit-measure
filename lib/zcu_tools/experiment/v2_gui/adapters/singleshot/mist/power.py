from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)

from .._shared import read_ge_centers, readout_probe_freq

MistPowerRunResult: TypeAlias = PowerResult


@dataclass
class MistPowerAnalyzeResult(FigureOnlyAnalyzeResult):
    # MIST gain sweep is look-at-the-curve: the domain analyze renders the
    # population-vs-gain (or vs photon-number when ac_coeff is known) traces and
    # extracts no scalar, so there is no writeback. ``figure`` is inherited.
    pass


class MistPowerAdapter(
    BaseAdapter[PowerCfg, MistPowerRunResult, MistPowerAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-power sweep: drives a probe pulse whose gain is swept "
            "and classifies each shot in-program against the |g>/|e> "
            "single-shot centres, plotting the ground/excited/other "
            "populations versus probe gain (or photon number when an AC-Stark "
            "coefficient is known). Runs on real hardware; the result is "
            "already populations (no per-point fit)."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; the run "
            "classifies each shot against them and fast-fails if any is "
            "missing. Optionally reads 'confusion_matrix' (readout correction) "
            "and 'ac_stark_coeff' (rescales the x-axis to photon number) at "
            "analyze time, and 't1' to set the relax delay; 'readout_f' or "
            "'r_f' plus 'res_ch' seed the probe drive."
        ),
        expects_ml=(
            "Needs a probe pulse and a readout module. Optionally references a "
            "calibrated reset and an init pulse — both disabled when no library "
            "entry exists."
        ),
        typical_writeback=(
            "No writeback — the population curves are read off the plot by eye."
        ),
        recommended=(
            "Run after 'singleshot/ge' has calibrated the discrimination. Sweep "
            "the probe gain across the MIST onset; provide 'ac_stark_coeff' "
            "(from the AC-Stark experiment) for a photon-number x-axis."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors PowerModuleCfg: reset, init_pulse,
            # probe_pulse, readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_readout_module_spec(),
            },
            sweep={"gain": SweepSpec(label="Probe gain (a.u.)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=1000, rounds=100, relax_delay=proper_relax(ctx, fallback=30.5)
            )
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", Init.DISABLED)
            .role("modules.init_pulse", "pi_pulse", Init.DISABLED)
            .role("modules.probe_pulse", "res_probe", Init.INLINE)
            .set("modules.probe_pulse.freq", readout_probe_freq(ctx))
            .role("modules.readout", "readout")
            .set_sweep("sweep.gain", SweepValue(start=0.0, stop=1.0, expts=151))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistPowerRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return PowerExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[MistPowerRunResult, NoAnalyzeParams]
    ) -> MistPowerAnalyzeResult:
        # ``ac_coeff`` (= md 'ac_stark_coeff', the AC-Stark experiment's product),
        # ``log_scale`` and ``confusion_matrix`` are analyze inputs read from md,
        # not user knobs; absent → the domain defaults (linear gain x-axis, no
        # readout correction).
        ac_coeff = req.md.get("ac_stark_coeff")
        log_scale = bool(req.md.get("log_scale", False))
        confusion = req.md.get("confusion_matrix")
        fig = PowerExp().analyze(
            req.run_result,
            ac_coeff=ac_coeff,
            log_scale=log_scale,
            confusion_matrix=confusion,
        )
        return MistPowerAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_mist_power_{time.strftime('%m%d')}"
