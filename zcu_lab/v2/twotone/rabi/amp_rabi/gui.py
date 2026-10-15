from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
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

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

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
