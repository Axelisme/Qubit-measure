from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist import FreqCfg, FreqDepExp, FreqResult
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
)

from .._shared import read_ge_centers, readout_probe_freq, readout_probe_freq_range

MistFreqRunResult: TypeAlias = FreqResult


@dataclass
class MistFreqAnalyzeResult(FigureOnlyAnalyzeResult):
    # MIST freq sweep is look-at-the-curve: the domain analyze renders the
    # ground/excited/other population-vs-frequency traces and extracts no scalar,
    # so there is no writeback. The single ``figure`` field is inherited.
    pass


class MistFreqAdapter(
    BaseAdapter[FreqCfg, MistFreqRunResult, MistFreqAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = FreqDepExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-frequency sweep: drives a probe pulse whose frequency "
            "is swept and classifies each shot in-program against the |g>/|e> "
            "single-shot centres, plotting the ground/excited/other populations "
            "versus probe frequency. Runs on real hardware; the result is "
            "already populations (no per-point fit)."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; the run "
            "classifies each shot against them and fast-fails if any is "
            "missing. Optionally reads 'confusion_matrix' (the GE 3x3 matrix) "
            "to readout-correct the populations at analyze time, and 't1' to "
            "set the relax delay; 'readout_f' or 'r_f' plus 'rf_w' / 'res_ch' "
            "seed the probe drive and frequency sweep."
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
            "Run after 'singleshot/ge' has calibrated the discrimination. A "
            "frequency span around the qubit drive captures the MIST response; "
            "use a large enough shot count (reps) for clean populations."
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
                "freq",
                label="Probe freq (MHz)",
                default=custom(
                    lambda ctx: readout_probe_freq_range(ctx, 51),
                    description="readout probe frequency range",
                ),
            )
            .reps(10000)
            .rounds(1)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistFreqRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return FreqDepExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[MistFreqRunResult, NoAnalyzeParams]
    ) -> MistFreqAnalyzeResult:
        # ``confusion_matrix`` (the GE 3x3 matrix) is an analyze input read from
        # md, not a user knob; absent → None, which skips the readout correction
        # (the domain default).
        confusion = req.md.get("confusion_matrix")
        fig = FreqDepExp().analyze(req.run_result, confusion_matrix=confusion)
        return MistFreqAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_mist_freq_{time.strftime('%m%d')}"
