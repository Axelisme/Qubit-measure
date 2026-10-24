from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
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


class AmplitudeRabiSummary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    pi_gain: float | None = Field(
        description="Owner pi_gain in gain; null retains an unavailable estimate."
    )

    pi_gain_err: float | None = Field(
        description="Owner pi_gain_err in gain; null retains an unavailable estimate."
    )

    pi2_gain: float | None = Field(
        description="Owner pi2_gain in gain; null retains an unavailable estimate."
    )

    pi2_gain_err: float | None = Field(
        description="Owner pi2_gain_err in gain; null retains an unavailable estimate."
    )


@dataclass
class AmpRabiAnalyzeResult(AnalyzeResultBase):
    # Fitted gains stay in this analysis result and the proposed pulse modules.
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
    fit_quality: dict[str, dict[str, object]] | None = None

    def summary(self) -> AmplitudeRabiSummary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return AmplitudeRabiSummary(
            pi_gain=self.pi_gain,
            pi_gain_err=self.pi_gain_err,
            pi2_gain=self.pi2_gain,
            pi2_gain_err=self.pi2_gain_err,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {**super().to_summary_dict(), **self.summary().model_dump()}


def _AmpRabi_make_form(ctx: FormContext) -> Form[AmpRabiCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _AmpRabi_analyze(
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


def _AmpRabi_get_writeback_items(
    req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalyzeResult],
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
    pulse = snapshot.qub_pulse
    length = pulse.waveform.length
    if not isinstance(length, (int, float)):
        raise ValueError("Amplitude Rabi pi_product requires a scalar source length")
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
        items.append(
            module_writeback(
                pulse,
                target=target,
                description=description,
                field_updates=(("gain", gain),),
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


DEFINITION = Adapter(
    spec=AmpRabiExp.AXES_SPEC,
    make_form=_AmpRabi_make_form,
    run=AmpRabiExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AmpRabi_analyze,
        options=lambda source, env: AmpRabiAnalyzeParams(),
        writeback=_AmpRabi_get_writeback_items,
    ),
)
