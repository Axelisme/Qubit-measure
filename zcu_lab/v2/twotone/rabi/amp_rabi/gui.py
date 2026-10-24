from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalysis,
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)

from .form import make_form

AmpRabiRunResult: TypeAlias = RunRecord[AmpRabiCfg, AmpRabiResult]


class AmplitudeRabiSummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    pi_amp: float | None = Field(
        description="Owner pi_amp in gain; null retains an unavailable estimate."
    )

    pi_amp_err: float | None = Field(
        description="Owner pi_amp_err in gain; null retains an unavailable estimate."
    )

    pi2_amp: float | None = Field(
        description="Owner pi2_amp in gain; null retains an unavailable estimate."
    )

    pi2_amp_err: float | None = Field(
        description="Owner pi2_amp_err in gain; null retains an unavailable estimate."
    )


@dataclass
class AmpRabiAnalyzeParams:
    skip: Annotated[int, ParamMeta(label="Skip points")] = 0


def _AmpRabi_analyze(
    req: AnalyzeRequest[AmpRabiRunResult, AmpRabiAnalyzeOptions], *, plots: Plots
) -> AmpRabiAnalysis:
    return AmpRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _AmpRabi_get_writeback_items(
    req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalysis],
) -> Sequence[WritebackItem]:
    """Propose five independent calibrations from the captured source.

    req pairs the source RunRecord, fitted gains and captured ctx.parameters.
    With a cfg, return pulse-module proposals at qubit.pulses.pi_amp/pi2_amp,
    separate x180/x90 slot proposals containing their resolved library keys,
    and qubit.ext.pi_product = pi_amp * source waveform.length (a.u. * us).
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
        ("$qubit.pulses.pi_amp", "$qubit.module.x180", result.pi_amp, "Pi pulse"),
        (
            "$qubit.pulses.pi2_amp",
            "$qubit.module.x90",
            result.pi2_amp,
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
            proposed_value=result.pi_amp * length,
        )
    )
    return items


def _AmpRabi_make_form(ctx: FormContext) -> Form[AmpRabiCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(
    cfg: AmpRabiCfg, *, context: RunContext
) -> RunRecord[AmpRabiCfg, AmpRabiResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=AmpRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AmpRabiExp.AXES_SPEC,
    make_form=_AmpRabi_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_AmpRabi_analyze,
        options=lambda source, env: AmpRabiAnalyzeOptions(),
        writeback=_AmpRabi_get_writeback_items,
    ),
)
