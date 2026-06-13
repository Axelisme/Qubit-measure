from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist import FreqCfg, FreqDepExp
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_qub_freq_range,
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
    require_soc_handles,
)

from .._shared import read_ge_centers

MistFreqRunResult: TypeAlias = Any  # FreqResult (frozen domain dataclass)


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

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
                "set the relax delay; 'q_f' / 'qub_ch' seed the probe drive."
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
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors FreqModuleCfg: reset, init_pulse,
            # probe_pulse, readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_readout_module_spec(),
            },
            sweep={"freq": SweepSpec(label="Probe freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=10000, rounds=1)
            .set("relax_delay", proper_relax(ctx))
            .role("modules.probe_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.freq", proper_qub_freq_range(ctx, 51))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistFreqRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
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
