from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitResult
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_lab.v2.singleshot.len_rabi.form import FORM

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
    experiment_tag: ClassVar[str | None] = LenRabiExp.persistence_tag()
    form = FORM
    exp_cls = LenRabiExp
    ExpCfg_cls = LenRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot Length Rabi: sweeps the qubit-drive pulse length, preserves "
            "every raw IQ shot, and derives ground / excited / other population curves "
            "during live view and analysis. Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay starts at 50.5 us. Length sweep is 0.03 us to 4 * "
            "$qubit.module.x180.waveform.length. Inline qubit gain starts at 1.0. "
            "Classification seeds $qubit.ext.readout_cal.g_center/e_center/radius; "
            "missing calibration stays Invalid. Inline templates read $qubit.freq, "
            "$qubit.wiring.L01.ch and $resonator.freq/wiring.ch/wiring.ro_ch; readout "
            "trigger uses $resonator.ext.time_of_fly + 0.05 us. Missing required sources "
            "remain Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Qubit pulse starts from qub_probe. Reset uses optional $qubit.module.reset; "
            "readout uses $qubit.module.readout or readout template. Sweep-overwritten "
            "pulse values remain editable and library keys stay selectable. Run uses this "
            "fixed capture."
        ),
        typical_writeback=(
            "When the joint fit is valid and its complete calibration is finite, proposes "
            "g_center, e_center, ge_radius, and confusion_matrix as four independent "
            "items. The four-item proposal is all-or-none."
        ),
        recommended=(
            "Set Initial State to the predominant state before the swept drive pulse even "
            "when the first sweep point is nonzero. Fit phase offset allows a correction "
            "within +/-90 degrees of that initial-state direction; the extrapolated "
            "zero-length population can then differ from the pre-pulse population. It is "
            "disabled by default. Set calibration cfg directly or seed it with "
            "'singleshot/ge'. A sweep spanning a few pi lengths captures a full "
            "oscillation. Review the measured population curves and overlaid joint-fit "
            "curves before applying all four calibration proposals."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
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
