from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)
from zcu_tools.experiment.v2.singleshot.rabi_fit import RabiJointFitResult
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ParamMeta,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.cfg import EvalValue, ScalarSpec

# ``AmpRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *gain* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsAmpRabiRunResult: TypeAlias = AmpRabiResult


@dataclass
class SsAmpRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"


@dataclass
class SsAmpRabiAnalyzeResult(AnalyzeResultBase):
    # The full numeric fit is intentionally non-JSON-safe and therefore omitted
    # from the GUI summary. The operator reviews the population/fit Figure while
    # writeback projection reads the typed domain result directly.
    fit_result: RabiJointFitResult
    figure: Figure


class SsAmpRabiAdapter(
    BaseAdapter[
        AmpRabiCfg, SsAmpRabiRunResult, SsAmpRabiAnalyzeResult, SsAmpRabiAnalyzeParams
    ]
):
    exp_cls = AmpRabiExp
    ExpCfg_cls: ClassVar[Any] = AmpRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot amp Rabi: sweeps the qubit-drive pulse gain, "
            "preserves every raw IQ shot, and derives ground / excited / other "
            "population curves during live view and analysis. Runs on real hardware."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. Missing or invalid cfg "
            "calibration fails before hardware. These values support live classification; "
            "the saved raw-IQ analysis jointly refits its calibration. Reads 'pi_gain' "
            "to seed the gain sweep stop (4*pi_gain when calibrated; fallback "
            "sweep -0.3 to 1.0 a.u.); "
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
            "(at zero length/gain), even when the first sweep point is nonzero. "
            "Set calibration cfg directly or seed it with 'singleshot/ge'. "
            "A sweep spanning a few pi gains "
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
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=-0.3,
                    stop=scaled_md("pi_gain", factor=4.0, fallback_value=1.0),
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

    def analyze(
        self, req: AnalyzeRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeParams]
    ) -> SsAmpRabiAnalyzeResult:
        fit_result, figure = AmpRabiExp().analyze(
            req.run_result, initial_state=req.analyze_params.initial_state
        )
        return SsAmpRabiAnalyzeResult(fit_result=fit_result, figure=figure)

    def get_writeback_items(
        self, req: WritebackRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        fit = req.analyze_result.fit_result
        calibration_is_finite = (
            fit.backend.valid
            and np.isfinite([fit.g_center.real, fit.g_center.imag]).all()
            and np.isfinite([fit.e_center.real, fit.e_center.imag]).all()
            and np.isfinite(fit.radius)
            and np.isfinite(fit.confusion_matrix).all()
        )
        if not calibration_is_finite:
            return []

        return [
            MetaDictWriteback(
                target_name="g_center",
                description="Amp Rabi fitted |g> IQ cluster centre (complex)",
                proposed_value=fit.g_center,
            ),
            MetaDictWriteback(
                target_name="e_center",
                description="Amp Rabi fitted |e> IQ cluster centre (complex)",
                proposed_value=fit.e_center,
            ),
            MetaDictWriteback(
                target_name="ge_radius",
                description="Amp Rabi fitted single-shot classification radius",
                proposed_value=fit.radius,
            ),
            MetaDictWriteback(
                target_name="confusion_matrix",
                description="Amp Rabi fitted 3x3 confusion matrix",
                proposed_value=fit.confusion_matrix.tolist(),
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_amp_rabi_{time.strftime('%m%d')}"
