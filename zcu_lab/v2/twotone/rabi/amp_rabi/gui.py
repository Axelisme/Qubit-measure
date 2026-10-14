from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)

from .form import make_form

AmpRabiRunResult: TypeAlias = RunRecord[AmpRabiCfg, AmpRabiResult]


@dataclass
class AmpRabiAnalyzeParams:
    skip: Annotated[int, ParamMeta(label="Skip points")] = 0


@dataclass
class AmpRabiAnalyzeResult(AnalyzeResultBase):
    # Fitted gains stay in this analysis result and the proposed pulse modules.
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
    fit_quality: dict[str, dict[str, object]] | None = None


class AmpRabiAdapter(
    BaseAdapter[
        AmpRabiCfg,
        AmpRabiRunResult,
        AmpRabiAnalyzeResult,
        AmpRabiAnalyzeParams,
    ]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[AmpRabiCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = AmpRabiExp.persistence_tag()
    exp_cls = AmpRabiExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Amplitude Rabi: drives the qubit at its known frequency and sweeps the "
            "drive-pulse gain, fitting the resulting Rabi oscillation to find the pi "
            "and pi/2 pulse amplitudes. Runs on real hardware. Run once you know the "
            "qubit frequency, to calibrate a pi pulse by amplitude (the pulse length "
            "stays fixed)."
        ),
        expects_md=(
            "Gain runs -0.3 to 1.2 * qubit.module.x180.gain over 51 points; Run "
            "overwrites drive gain. Relax delay is 5 * qubit.t1 us. Missing calibration "
            "leaves the affected fields empty and Invalid; no numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). The drive uses "
            "qubit.module.x180 with a pi_pulse template fallback. Unset or dangling "
            "slots use the named template, not a legacy blank. Key choices remain "
            "available. Inline templates read qubit.freq, qubit.wiring.L01.ch, "
            "resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + "
            "0.05 us; missing required calibration remains Invalid."
        ),
        typical_writeback=(
            "Copies the source drive pulse into captured qubit.pulses.pi_amp and "
            "pi2_amp, replacing gain with the fitted pi and pi/2 gains. Separately "
            "proposes qubit.module.x180 and x90 pointing to those resolved library "
            "keys, even when already bound to the same keys. Also proposes "
            "qubit.ext.pi_product as fitted pi gain times the source pulse length (a.u. "
            "* us). Every item is independently selectable; none is proposed without a "
            "source cfg. Does not write separate gain scalars or read current "
            "calibration."
        ),
        recommended=(
            "Analysis takes a 'Skip points' count (default 0) to drop leading low-gain "
            "sweep points before the cosine fit; raise it only if the first few points "
            "are distorted. A gain sweep spanning roughly two pi amplitudes captures at "
            "least a full oscillation; widen it if no full period is visible."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiRunResult:
        """Build typed cfg from resolved GUI values and run amplitude Rabi.

        req provides the captured cfg context and device snapshot; context owns
        acquisition and cancellation. Return the typed cfg alongside the result.
        Config, device and acquisition failures propagate without live defaults.
        """
        del req
        result = AmpRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[AmpRabiRunResult, AmpRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> AmpRabiAnalyzeResult:
        params = req.analyze_params
        analysis = AmpRabiExp().analyze(
            req.run_result,
            AmpRabiAnalyzeOptions(skip=params.skip),
            plots=plots,
        )
        return AmpRabiAnalyzeResult(
            pi_gain=analysis.pi_amp,
            pi_gain_err=analysis.pi_amp_err,
            pi2_gain=analysis.pi2_amp,
            pi2_gain_err=analysis.pi2_amp_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
        )

    def get_writeback_items(
        self, req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose five independent calibrations from the captured source.

        req pairs the source RunRecord, fitted gains and captured ctx.parameters.
        With a cfg, return pulse-module proposals at qubit.pulses.pi_amp/pi2_amp,
        separate x180/x90 slot proposals containing their resolved library keys,
        and qubit.ext.pi_product = pi_gain * source waveform.length (a.u. * us).
        Same-key slots are still proposed. Without cfg return no items.
        Only module gains change; source cfg is not mutated. Missing capture,
        role resolution, non-scalar source length and conversion/asset faults
        fail explicitly. No hardware, live calibration reads or writes occur.
        """
        snapshot = req.run_result.cfg
        if snapshot is None:
            return []
        parameters = req.ctx.parameters
        if parameters is None:
            raise ValueError("Amplitude Rabi writeback requires captured parameters")
        pulse = snapshot.modules.qub_pulse
        length = pulse.waveform.length
        if not isinstance(length, (int, float)):
            raise ValueError(
                "Amplitude Rabi pi_product requires a scalar source length"
            )
        result = req.analyze_result
        items: list[WritebackItem] = []
        for target, slot, gain, description in (
            ("$qubit.pulses.pi_amp", "$qubit.module.x180", result.pi_gain, "Pi pulse"),
            (
                "$qubit.pulses.pi2_amp",
                "$qubit.module.x90",
                result.pi2_gain,
                "Pi/2 pulse",
            ),
        ):
            key = parameters.library_key(target)
            spec, value = module_cfg_to_value(pulse)
            value.with_field("gain", gain)
            items.append(
                ModuleWriteback(
                    target_name=target,
                    description=description,
                    edit_schema=CfgSchema(spec=spec, value=value),
                )
            )
            items.append(
                ParameterWriteback(
                    target_name=slot,
                    description=f"{description} slot",
                    proposed_value=key,
                )
            )
        items.append(
            ParameterWriteback(
                target_name="$qubit.ext.pi_product",
                description="Pi pulse gain times length (a.u. * us)",
                proposed_value=result.pi_gain * length,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_amp_rabi_{time.strftime('%m%d')}"
