from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_tools.experiment.v2.singleshot.rabi_fit import RabiJointFitResult
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
)

from ._rabi import rabi_calibration_writeback
from ._shared import read_ge_centers

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsLenRabiRunResult: TypeAlias = LenRabiResult


@dataclass
class SsLenRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


@dataclass
class SsLenRabiAnalyzeResult(AnalyzeResultBase):
    # The full numeric fit is intentionally non-JSON-safe and therefore omitted
    # from the GUI summary. The operator reviews the population/fit Figure while
    # writeback projection reads the typed domain result directly.
    fit_result: RabiJointFitResult
    figure: Figure


class SsLenRabiAdapter(
    BaseAdapter[
        LenRabiCfg, SsLenRabiRunResult, SsLenRabiAnalyzeResult, SsLenRabiAnalyzeParams
    ]
):
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot Length Rabi: sweeps the qubit-drive pulse length, "
            "preserves every raw IQ shot, and derives ground / excited / other "
            "population curves during live view and analysis. Runs on real hardware."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run "
            "fast-fails if any is missing. Those values support live classification; "
            "the saved raw-IQ analysis jointly refits its calibration. Reads 'pi_len' "
            "to seed the sweep stop (4*pi_len when calibrated; fallback sweep "
            "0.03–0.2 us); "
            "'q_f' / 'qub_ch' to seed the qubit-drive defaults."
        ),
        expects_ml=(
            "Needs a qubit drive-pulse module (qub_pulse) and a readout module. "
            "Optional reset (disabled when no library entry exists)."
        ),
        typical_writeback=(
            "When the joint fit is valid and its complete calibration is finite, "
            "proposes g_center, e_center, ge_radius, and confusion_matrix as four "
            "independent items. The four-item proposal is all-or-none."
        ),
        recommended=(
            "Set Initial State to the predominant state before the swept drive pulse "
            "even when the first sweep point is nonzero. Fit phase offset "
            "allows a correction within +/-90 degrees of that initial-state "
            "direction; the extrapolated zero-length population can then differ "
            "from the pre-pulse population. It is disabled by default. "
            "Run after 'singleshot/ge'. A sweep spanning a few pi lengths "
            "captures a full oscillation. Review the measured population curves "
            "and overlaid joint-fit curves before applying all four calibration "
            "proposals."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                overrides={"gain": 1.0},
            )
            .readout()
            .relax_delay(50.5)
            .sweep(
                "length",
                label="Length (us)",
                default=SweepDefault(
                    start=0.03,
                    stop=scaled_md("pi_len", factor=4.0, fallback_value=0.2),
                    expts=51,
                ),
            )
            .int("shots", label="Shots", default=1000)
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> SsLenRabiRunResult:
        # Override standard run: domain run needs the GE classification trio.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return LenRabiExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    def analyze(
        self, req: AnalyzeRequest[SsLenRabiRunResult, SsLenRabiAnalyzeParams]
    ) -> SsLenRabiAnalyzeResult:
        fit_result, figure = LenRabiExp().analyze(
            req.run_result,
            decay=req.analyze_params.decay,
            fit_phase=req.analyze_params.fit_phase,
            initial_state=req.analyze_params.initial_state,
        )
        return SsLenRabiAnalyzeResult(fit_result=fit_result, figure=figure)

    def get_writeback_items(
        self, req: WritebackRequest[SsLenRabiRunResult, SsLenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_len_rabi_{time.strftime('%m%d')}"
