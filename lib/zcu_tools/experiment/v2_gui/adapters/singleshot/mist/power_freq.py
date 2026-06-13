from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.mist import FreqPowerCfg, FreqPowerExp
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
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

from .._shared import read_ge_centers

MistPowerFreqRunResult: TypeAlias = Any  # FreqPowerResult (frozen domain dataclass)


@dataclass
class MistPowerFreqAnalyzeResult(FigureOnlyAnalyzeResult):
    # MIST 2D landscape is look-at-the-image: the domain analyze renders the
    # ground/excited/other population maps over (gain, freq) and extracts no
    # scalar, so there is no writeback. ``figure`` is inherited.
    pass


class MistPowerFreqAdapter(
    BaseAdapter[
        FreqPowerCfg,
        MistPowerFreqRunResult,
        MistPowerFreqAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = FreqPowerExp
    ExpCfg_cls: ClassVar[Any] = FreqPowerCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "MIST probe power × frequency landscape: drives a probe pulse while "
                "sweeping its gain and frequency (2D), classifying each shot "
                "in-program against the |g>/|e> single-shot centres, and plotting "
                "the ground/excited/other populations as 2D maps over (gain, freq). "
                "Runs on real hardware; the result is already populations (no "
                "per-point fit)."
            ),
            expects_md=(
                "REQUIRES the single-shot discrimination calibration in the "
                "MetaDict — run 'singleshot/ge' first and apply its writeback so "
                "'g_center' / 'e_center' / 'ge_radius' are present; the run "
                "classifies each shot against them and fast-fails if any is "
                "missing. Optionally reads 'confusion_matrix' (readout correction), "
                "'ac_stark_coeff' and 'log_scale' at analyze time; 't1' to set the "
                "relax delay; 'q_f' / 'qub_ch' seed the probe drive."
            ),
            expects_ml=(
                "Needs a probe pulse and a readout module. Optionally references a "
                "calibrated reset and an init pulse — both disabled when no library "
                "entry exists."
            ),
            typical_writeback=(
                "No writeback — the population landscapes are read off the plot by eye."
            ),
            recommended=(
                "Run after 'singleshot/ge' has calibrated the discrimination. Sweep "
                "the probe gain across the MIST onset and the frequency around the "
                "qubit/resonator line of interest."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors FreqPowerModuleCfg: reset, init_pulse,
            # probe_pulse, readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_readout_module_spec(),
            },
            # 2D sweep: probe frequency (outer in cfg) × probe gain (inner scan).
            sweep={
                "freq": SweepSpec(label="Probe frequency (MHz)"),
                "gain": SweepSpec(label="Probe gain (a.u.)"),
            },
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
            .set_sweep("sweep.freq", SweepValue(start=-100.0, stop=100.0, expts=101))
            .set_sweep("sweep.gain", SweepValue(start=0.0, stop=1.0, expts=51))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> MistPowerFreqRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return FreqPowerExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[MistPowerFreqRunResult, NoAnalyzeParams]
    ) -> MistPowerFreqAnalyzeResult:
        # ``ac_coeff`` (= md 'ac_stark_coeff'), ``log_scale`` and
        # ``confusion_matrix`` are analyze inputs read from md, not user knobs;
        # absent → domain defaults (linear axes, no readout correction).
        ac_coeff = req.md.get("ac_stark_coeff")
        log_scale = bool(req.md.get("log_scale", False))
        confusion = req.md.get("confusion_matrix")
        fig = FreqPowerExp().analyze(
            req.run_result,
            ac_coeff=ac_coeff,
            log_scale=log_scale,
            confusion_matrix=confusion,
        )
        return MistPowerFreqAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_mist_power_freq_{time.strftime('%m%d')}"
