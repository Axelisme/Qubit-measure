from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SampleWriteback,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)

from .form import FORM

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = RunRecord[T2RamseyCfg, T2RamseyResult]


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


@dataclass
class T2RamseyAnalyzeResult(AnalyzeResultBase):
    t2r: float
    t2r_err: float
    # ``detune`` is the fitted fringe frequency (MHz); meaningful only when the
    # fringe fit ran (``fit_fringe`` True). Decay-only fits report 0.0.
    detune: float
    fit_fringe: bool
    fit_quality: dict[str, dict[str, object]] | None = None


class T2RamseyAdapter(
    BaseAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    form = FORM
    experiment_tag: ClassVar[str | None] = T2RamseyExp.persistence_tag()
    exp_cls = T2RamseyExp
    ExpCfg_cls: ClassVar[Any] = T2RamseyCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T2 Ramsey: two pi/2 pulses separated by a swept free-evolution delay; fits "
            "the decaying fringe to extract the Ramsey coherence time T2* and the "
            "qubit-drive detuning. Runs on real hardware. Run after a pi/2 pulse is "
            "calibrated; a small deliberate detuning produces the fringe whose "
            "frequency reads out the true qubit frequency offset."
        ),
        expects_md=(
            "Delay runs 0 to 1.5 * qubit.t2r us over 101 points. Detune ratio is "
            "fringes per sweep step; native MHz detune is ratio / step. Relax delay is "
            "5 * qubit.t1 us. Missing calibration leaves the affected fields empty and "
            "Invalid; no numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). The drive uses "
            "qubit.module.x90 with a pi2_pulse template fallback. Unset or dangling "
            "slots use the named template, not a legacy blank. Key choices remain "
            "available. Inline templates read qubit.freq, qubit.wiring.L01.ch, "
            "resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + "
            "0.05 us; missing required calibration remains Invalid."
        ),
        typical_writeback=(
            "Proposes fitted qubit.t2r (us) to the captured qubit and working point. A "
            "fringe fit with source cfg and known realized detune also proposes "
            "qubit.freq (MHz): source pi/2 frequency + realized detune - fitted fringe "
            "detune. A pure decay fit or a loaded source without realized detune "
            "proposes only qubit.t2r. No module writeback; later focus changes do not "
            "retarget it. An independent t2r_us/t2r_err_us sample uses the captured Run "
            "bias when device evidence is available."
        ),
        recommended=(
            "Enable Fit phase offset to fit a free fringe phase; it defaults to fixed "
            "phase and has no effect for decay-only analysis. Set the 'Detune ratio "
            "(fringes/step)' cfg knob to a small deliberate offset (default 0.05) — it "
            "is the number of fringe periods per delay-sweep step; the absolute detune "
            "(MHz) applied to the second pi/2 pulse phase is detune_ratio / sweep step, "
            "so the Ramsey signal oscillates at that detuning. The analysis 'Fit "
            "fringe' option (on by default, matching the nonzero default ratio) fits "
            "the fringe frequency to read back the true qubit-frequency offset and "
            "write qubit.freq. Set the ratio to 0 and turn 'Fit fringe' off for a pure "
            "decay fit when driving on resonance. A delay sweep reaching a few T2* "
            "captures enough fringe periods and decay."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> T2RamseyRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = self.exp_cls().run(cfg, context=context)
        logger.info("T2Ramsey true detune: %s MHz", result.true_activate_detune)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeParams],
        *,
        plots: Plots,
    ) -> T2RamseyAnalyzeResult:
        params = req.analyze_params
        analysis = T2RamseyExp().analyze(
            req.run_result,
            T2RamseyAnalyzeOptions(
                fit_fringe=params.fit_fringe, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        return T2RamseyAnalyzeResult(
            t2r=analysis.t2r,
            t2r_err=analysis.t2r_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
            detune=analysis.detune,
            fit_fringe=params.fit_fringe,
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose captured qubit.t2r, plus qubit.freq for a sourced fringe fit.

        req supplies the committed fit, source record and captured context.
        Frequency is source pi/2 frequency + realized detune - fitted detune;
        it is omitted for decay fits, cfg=None or unknown realized detune.
        Non-scalar source frequency raises ValueError. WritebackService resolves
        destinations on req.ctx.parameters and rejects missing captures/unbound
        roles. With a capture, also propose t2r_us/t2r_err_us for the Run's
        resolved $qubit; an unbound alias raises RoleResolutionError. Without
        a capture, retain parameter proposals only so loaded data can still be
        analyzed. The service adds bias coordinates or omits the sample with a
        public reason. This Run leaves devices fixed; no live calibration lookup,
        device access or write occurs here.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t2r",
                description="T2 Ramsey time (us)",
                proposed_value=result.t2r,
            ),
        ]

        # Frequency correction uses only the source run, never current calibration.
        snapshot = req.run_result.cfg
        if (
            result.fit_fringe
            and snapshot is not None
            and req.run_result.result.true_activate_detune is not None
        ):
            source_freq = snapshot.modules.pi2_pulse.freq
            if not isinstance(source_freq, float):
                raise ValueError(
                    "qubit.freq writeback requires a scalar source pi/2 frequency"
                )
            q_f = (
                source_freq + req.run_result.result.true_activate_detune - result.detune
            )
            items.append(
                ParameterWriteback(
                    target_name="$qubit.freq",
                    description="Qubit frequency (MHz)",
                    proposed_value=q_f,
                )
            )

        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="T2 Ramsey sample at the captured bias",
                    proposed_value={
                        "t2r_us": result.t2r,
                        "t2r_err_us": result.t2r_err,
                    },
                )
            )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t2ramsey_{time.strftime('%m%d')}"
