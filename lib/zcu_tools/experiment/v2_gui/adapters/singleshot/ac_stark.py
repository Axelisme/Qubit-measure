from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot import AcStarkCfg, AcStarkExp
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)

from ._shared import read_chi_kappa, read_ge_centers

# Domain AcStarkExp.analyze returns (ac_coeff, fig). The fitted AC-Stark
# coefficient is written back to the MetaDict (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = Any  # AcStarkResult (frozen domain dataclass)


@dataclass
class SsAcStarkAnalyzeResult(AnalyzeResultBase):
    ac_stark_coeff: float
    figure: Figure


class SsAcStarkAdapter(
    BaseAdapter[
        AcStarkCfg,
        SsAcStarkRunResult,
        SsAcStarkAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = AcStarkExp
    ExpCfg_cls: ClassVar[Any] = AcStarkCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Single-shot AC-Stark calibration: drives two Stark pulses while "
                "sweeping the first pulse's gain and the probe frequency (2D), "
                "classifying each shot in-program against the |g>/|e> IQ-cluster "
                "centres. Fits the Stark-shifted resonance versus gain² to extract "
                "the AC-Stark coefficient (photon number per gain²). Runs on real "
                "hardware."
            ),
            expects_md=(
                "REQUIRES the single-shot discrimination calibration in the "
                "MetaDict — run 'singleshot/ge' first and apply its writeback so "
                "'g_center' / 'e_center' / 'ge_radius' are present; the run "
                "classifies each shot against them and fast-fails if any is missing. "
                "ANALYSIS additionally REQUIRES 'chi' (dispersive shift, MHz) and "
                "'rf_w' (resonator linewidth kappa, MHz) — both feed the AC-Stark "
                "coefficient fit and analyze fast-fails if either is missing (run "
                "the dispersive-shift experiment first). Optionally reads "
                "'confusion_matrix' (readout correction) and 'cutoff' (drop gains "
                "above it before fitting) at analyze time."
            ),
            expects_ml=(
                "Needs two Stark pulses and a readout module. Optionally references "
                "a calibrated reset and an init pulse — both disabled when no "
                "library entry exists."
            ),
            typical_writeback=(
                "Proposes the fitted AC-Stark coefficient into MetaDict "
                "'ac_stark_coeff' (photon number per gain²); the MIST experiments "
                "read it to rescale their x-axis to photon number."
            ),
            recommended=(
                "Run after 'singleshot/ge' and after the dispersive-shift "
                "experiment has set 'chi' / 'rf_w'. Sweep the Stark gain across the "
                "onset and the probe frequency around the qubit line."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors AcStarkModuleCfg: reset, init_pulse,
            # stark_pulse1, stark_pulse2, readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "stark_pulse1": make_pulse_module_spec(label="Stark Pulse 1"),
                "stark_pulse2": make_pulse_module_spec(label="Stark Pulse 2"),
                "readout": make_readout_module_spec(),
            },
            # 2D sweep: Stark gain (outer) × probe frequency (inner).
            sweep={
                "gain": SweepSpec(label="Stark gain (a.u.)"),
                "freq": SweepSpec(label="Probe frequency (MHz)"),
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=2)
            .set("relax_delay", proper_relax(ctx))
            .role("modules.stark_pulse1", "qub_probe", prefer_blank=True)
            .role("modules.stark_pulse2", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.gain", SweepValue(start=0.0, stop=0.22, expts=301))
            .set_sweep("sweep.freq", SweepValue(start=-700.0, stop=100.0, expts=101))
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> SsAcStarkRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return AcStarkExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def analyze(
        self, req: AnalyzeRequest[SsAcStarkRunResult, NoAnalyzeParams]
    ) -> SsAcStarkAnalyzeResult:
        # ``chi`` / ``kappa`` (= md 'rf_w', the resonator linewidth) are required
        # fit inputs read from md — fast-fail if either is missing. The domain
        # derives eta = kappa²/(kappa²+chi²) internally. ``confusion_matrix``
        # (readout correction) and ``cutoff`` (drop high gains before fitting) are
        # optional md inputs, never user knobs; absent → domain defaults.
        chi, kappa = read_chi_kappa(req.md)
        confusion = req.md.get("confusion_matrix")
        cutoff = req.md.get("cutoff")
        ac_coeff, fig = AcStarkExp().analyze(
            chi,
            req.run_result,
            kappa=kappa,
            confusion_matrix=confusion,
            cutoff=cutoff,
        )
        return SsAcStarkAnalyzeResult(ac_stark_coeff=ac_coeff, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsAcStarkRunResult, SsAcStarkAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``ac_stark_coeff`` per single_qubit.md:3329.
        return [
            MetaDictWriteback(
                target_name="ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_stark_coeff,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sh_ac_stark_{time.strftime('%m%d')}"
