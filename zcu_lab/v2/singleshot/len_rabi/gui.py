from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import SweepDefault, scaled_md
from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2._support.singleshot.rabi_fit import RabiJointFitResult
from zcu_lab.v2.singleshot.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsLenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


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
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. These values support live classification; "
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
            "Set calibration cfg directly or seed it with 'singleshot/ge'. A sweep spanning a few pi lengths "
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
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("ge_radius"),
            )
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SsLenRabiRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LenRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsLenRabiRunResult, SsLenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsLenRabiAnalyzeResult:
        fit_result = LenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(
                decay=req.analyze_params.decay,
                fit_phase=req.analyze_params.fit_phase,
                initial_state=req.analyze_params.initial_state,
            ),
            plots=plots,
        )
        return SsLenRabiAnalyzeResult(fit_result=fit_result)

    def get_writeback_items(
        self, req: WritebackRequest[SsLenRabiRunResult, SsLenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_len_rabi_{time.strftime('%m%d')}"
