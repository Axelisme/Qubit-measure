from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    FigureOnlyAnalyzeResult,
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    SweepValue,
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse(
                "probe_pulse",
                role_id="res_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="readout probe frequency",
                    )
                },
            )
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=30.5))
            .sweep(
                "gain",
                label="Probe gain (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=151),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistPowerRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
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
