from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot import AcStarkCfg, AcStarkExp
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    md_has_key,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    EvalValue,
    SweepValue,
)

from ._shared import read_chi_kappa, read_ge_centers

# Domain AcStarkExp.analyze returns (ac_coeff, fig). The fitted AC-Stark
# coefficient is written back to the MetaDict (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = Any  # AcStarkResult (frozen domain dataclass)


def _freq_sweep_default(ctx: ExpContext) -> SweepValue:
    if md_has_key(ctx, "q_f"):
        return SweepValue(
            start=EvalValue(expr="q_f - 700.0"),
            stop=EvalValue(expr="q_f + 100.0"),
            expts=101,
        )
    return SweepValue(start=3300.0, stop=4100.0, expts=101)


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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse(
                "stark_pulse1",
                role_id="qub_probe",
                label="Stark Pulse 1",
                init=ModuleInit.INLINE,
            )
            .pulse(
                "stark_pulse2",
                role_id="qub_probe",
                label="Stark Pulse 2",
                init=ModuleInit.INLINE,
            )
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "gain",
                label="Stark gain (a.u.)",
                default=SweepValue(start=0.0, stop=0.22, expts=301),
            )
            .sweep(
                "freq",
                label="Probe frequency (MHz)",
                default=custom(
                    _freq_sweep_default,
                    description="AC-Stark probe frequency range",
                ),
            )
            .reps(1000)
            .rounds(2)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> SsAcStarkRunResult:
        # Override the standard run path: the domain run needs the GE
        # classification trio (not in cfg) — read it from md and forward it.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
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
