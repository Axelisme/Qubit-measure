from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist.pre_freq import PreFreqCfg, PreFreqExp
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

MistPreFreqRunResult: TypeAlias = Any  # PreFreqResult (frozen domain dataclass)


@dataclass
class MistPreFreqAnalyzeResult(FigureOnlyAnalyzeResult):
    # MIST pre-pulse frequency sweep is look-at-the-curve: the domain analyze
    # renders the population-vs-pre-pulse-frequency traces and extracts no scalar,
    # so there is no writeback. ``figure`` is inherited.
    pass


class MistPreFreqAdapter(
    BaseAdapter[
        PreFreqCfg, MistPreFreqRunResult, MistPreFreqAnalyzeResult, NoAnalyzeParams
    ]
):
    exp_cls = PreFreqExp
    ExpCfg_cls: ClassVar[Any] = PreFreqCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "MIST pre-pulse frequency sweep: applies a frequency-swept "
                "pre-pulse (the init pulse) before the probe, classifying each "
                "shot in-program against the |g>/|e> single-shot centres, plotting "
                "the ground/excited/other populations versus pre-pulse frequency. "
                "Runs on real hardware; the result is already populations."
            ),
            expects_md=(
                "REQUIRES the single-shot discrimination calibration in the "
                "MetaDict — run 'singleshot/ge' first and apply its writeback so "
                "'g_center' / 'e_center' / 'ge_radius' are present; the run "
                "classifies each shot against them and fast-fails if any is "
                "missing. Optionally reads 'confusion_matrix' (readout correction) "
                "at analyze time, and 't1' for the relax delay; 'q_f' / 'qub_ch' "
                "seed the pulse drives."
            ),
            expects_ml=(
                "Needs an init (pre-)pulse, a probe pulse and a readout module. "
                "Optionally references a calibrated reset and a pi pulse — both "
                "disabled when no library entry exists."
            ),
            typical_writeback=(
                "No writeback — the population curves are read off the plot by eye."
            ),
            recommended=(
                "Run after 'singleshot/ge' has calibrated the discrimination. Sweep "
                "the pre-pulse frequency across the spectral feature of interest."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors PreFreqModuleCfg: reset, init_pulse
            # (required — the swept pre-pulse), pi_pulse (optional), probe_pulse,
            # readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(label="Pre Pulse"),
                "pi_pulse": make_pulse_module_spec(label="Pi Pulse", optional=True),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_readout_module_spec(),
            },
            sweep={"freq": SweepSpec(label="Pre-pulse freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=10000, rounds=1)
            .set("relax_delay", proper_relax(ctx))
            .role("modules.init_pulse", "qub_probe", prefer_blank=True)
            .role("modules.probe_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.pi_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.freq", proper_qub_freq_range(ctx, 51))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistPreFreqRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return PreFreqExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[MistPreFreqRunResult, NoAnalyzeParams]
    ) -> MistPreFreqAnalyzeResult:
        # ``confusion_matrix`` is an analyze input read from md (not a user knob);
        # absent → None, which skips the readout correction (domain default).
        confusion = req.md.get("confusion_matrix")
        fig = PreFreqExp().analyze(req.run_result, confusion_matrix=confusion)
        return MistPreFreqAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_mist_pre_freq_{time.strftime('%m%d')}"
