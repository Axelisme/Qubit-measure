from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.time_domain.t2echo import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain._detune_shared import (
    detune_ratio_of,
    resolve_detune,
    strip_detune_ratio,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
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

logger = logging.getLogger(__name__)

T2EchoRunResult: TypeAlias = T2EchoResult


@dataclass
class T2EchoAnalyzeParams:
    fit_method: Annotated[Literal["fringe", "decay"], ParamMeta(label="Fit method")] = (
        "fringe"
    )


@dataclass
class T2EchoAnalyzeResult(AnalyzeResultBase):
    t2e: float
    t2e_err: float
    figure: Figure


class T2EchoAdapter(
    BaseAdapter[
        T2EchoCfg,
        T2EchoRunResult,
        T2EchoAnalyzeResult,
        T2EchoAnalyzeParams,
    ]
):
    exp_cls = T2EchoExp
    ExpCfg_cls: ClassVar[Any] = T2EchoCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T2 echo (spin/Hahn echo): a pi/2 – delay – pi – delay – pi/2 "
            "sequence with the total free-evolution delay swept; the "
            "refocusing pi pulse cancels low-frequency dephasing, and the "
            "decay fit yields the echo coherence time T2E. Runs on real "
            "hardware. Run after both pi and pi/2 pulses are calibrated; T2E "
            "is typically longer than the Ramsey T2*."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional, seeding defaults): 't2e' — "
            "prior T2-echo estimate (us); the total-delay sweep spans up to "
            "1.5*t2e (fallback ~20 us). 't1' seeds relax_delay (5*t1, fallback "
            "~100 us). The pi and pi/2 pulses pull 'q_f' (~2000–6000 MHz) and "
            "'qub_ch'. Readout pulls 'r_f' (~4000–8000 MHz), 'res_ch' / "
            "'ro_ch', and 'timeFly' for the readout trigger offset (~0–1 us)."
        ),
        expects_ml=(
            "Needs both a qubit pi-pulse module (prefers library 'pi_amp' / "
            "'pi_len') and a pi/2-pulse module (prefers 'pi2_amp' / 'pi2_len'); "
            "each falls back to a blank inline pulse. Plus a readout module "
            "(calibrated 'readout_dpm' / "
            "'readout_rf' / 'readout' / 'res_readout', else a blank "
            "pulse-readout referencing 'ro_waveform' when present). Optional "
            "reset references a library reset ('reset_bath' / 'reset_10' / "
            "'reset_120') when present."
        ),
        typical_writeback=(
            "Proposes the fitted T2-echo time into MetaDict 't2e' (us). No "
            "qubit-frequency writeback (unlike T2 Ramsey, echo does not update "
            "'q_f'). No ModuleLibrary writeback."
        ),
        recommended=(
            "Set the 'Detune ratio (fringes/step)' cfg knob (default 0.1) — it "
            "is the number of fringe periods per delay-sweep step; the absolute "
            "detune (MHz) applied to the final pi/2 pulse phase is detune_ratio "
            "/ sweep step, leaving a residual oscillation on the echo decay. "
            "Analysis 'Fit method' defaults to 'fringe' (matching the nonzero "
            "default ratio) so the fringe frequency is fit; the detune is "
            "computed but NOT written back. Set the ratio to 0 and switch 'Fit "
            "method' to 'decay' for a standard echo (pure exponential decay → "
            "T2E). A total-delay sweep reaching a few T2E lets the echo decay; "
            "the first sweep point is dropped in the fit."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse")
            .pulse("pi2_pulse", role_id="pi2_pulse")
            .readout()
            .relax_delay(
                scaled_md("t1", factor=5.0, fallback_value=100.0),
            )
            .sweep(
                "length",
                label="Total delay (us)",
                default=SweepDefault(
                    start=0.0,
                    stop=scaled_md("t2e", factor=1.5, fallback_value=20.0),
                    expts=101,
                ),
            )
            .float(
                "detune_ratio",
                label="Detune ratio (fringes/step)",
                default=0.1,
                decimals=3,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T2EchoCfg:
        # Strip the run-only detune_ratio knob before lowering to T2EchoCfg,
        # which would reject the unknown key.
        return req.ml.make_cfg(strip_detune_ratio(raw_cfg), T2EchoCfg)

    def run(self, req: RunRequest, schema: CfgSchema) -> T2EchoRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        # detune_ratio (fringes-per-step) → absolute applied detune (MHz) over the
        # lowered length sweep step (SweepCfg guarantees step != 0 for expts > 1).
        detune = resolve_detune(detune_ratio_of(raw_cfg), cfg.sweep.length.step)
        # T2EchoExp.run returns (result, true_detune); the GUI run contract
        # returns only the Result, so log the realized detune and drop it.
        result, true_detune = self.exp_cls().run(soc, soccfg, cfg, detune=detune)
        logger.info("T2 Echo true detune: %.3f MHz", true_detune)
        return result

    def analyze(
        self, req: AnalyzeRequest[T2EchoRunResult, T2EchoAnalyzeParams]
    ) -> T2EchoAnalyzeResult:
        params = req.analyze_params
        t2e, t2e_err, _, _, fig = T2EchoExp().analyze(
            req.run_result,
            fit_method=params.fit_method,
        )
        return T2EchoAnalyzeResult(t2e=t2e, t2e_err=t2e_err, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[T2EchoRunResult, T2EchoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="t2e",
                description="T2 Echo time (us)",
                proposed_value=result.t2e,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2echo_{time.strftime('%m%d')}"
