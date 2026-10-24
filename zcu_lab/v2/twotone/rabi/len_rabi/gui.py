from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
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
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalysis,
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)

from .form import make_form

LenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


class TimeRabiSummary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    pi_len: float | None = Field(
        description="Owner pi_len in us; null retains an unavailable estimate."
    )

    pi_len_err: float | None = Field(
        description="Owner pi_len_err in us; null retains an unavailable estimate."
    )

    pi2_len: float | None = Field(
        description="Owner pi2_len in us; null retains an unavailable estimate."
    )

    pi2_len_err: float | None = Field(
        description="Owner pi2_len_err in us; null retains an unavailable estimate."
    )

    rabi_f: float | None = Field(
        description="Owner rabi_f in MHz; null retains an unavailable estimate."
    )

    rabi_f_err: float | None = Field(
        description="Owner rabi_f_err in MHz; null retains an unavailable estimate."
    )


@dataclass
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


def _LenRabi_analyze(
    req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeOptions], *, plots: Plots
) -> LenRabiAnalysis:
    return LenRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def _LenRabi_get_writeback_items(
    req: WritebackRequest[LenRabiRunResult, LenRabiAnalysis],
) -> Sequence[WritebackItem]:
    """Propose independent calibrations from the source and captured roles.

    req pairs the source RunRecord, fitted lengths/frequency and captured
    ctx.parameters. Always propose qubit.ext.rabi_freq in MHz. With cfg,
    add qubit.pulses.pi_amp/pi2_amp pulse proposals with fitted lengths,
    independent x180/x90 slot proposals containing their resolved keys,
    and qubit.ext.pi_product = source gain * pi_len (a.u. * us).
    Same-key slots are still proposed. Without cfg return only rabi_freq.
    Only waveform lengths change; source cfg is not mutated. Missing
    capture, role resolution, non-scalar source gain and unsupported
    waveform-length conversion fail explicitly. No hardware, live
    calibration reads or writes occur.
    """
    parameters = req.ctx.parameters
    if parameters is None:
        raise ValueError("Length Rabi writeback requires captured parameters")
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.rabi_freq",
            description="Rabi oscillation frequency (MHz)",
            proposed_value=result.rabi_f,
        ),
    ]
    snapshot = req.run_result.cfg
    if snapshot is None:
        return items
    pulse = snapshot.qub_pulse
    gain = pulse.gain
    if not isinstance(gain, (int, float)):
        raise ValueError("Length Rabi pi_product requires a scalar source gain")
    for target, slot, length, description in (
        ("$qubit.pulses.pi_amp", "$qubit.module.x180", result.pi_len, "Pi pulse"),
        (
            "$qubit.pulses.pi2_amp",
            "$qubit.module.x90",
            result.pi2_len,
            "Pi/2 pulse",
        ),
    ):
        key = parameters.library_key(target)
        items.append(
            module_writeback(
                pulse,
                target=target,
                description=description,
                field_updates=(("waveform.length", length),),
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
            proposed_value=gain * result.pi_len,
        )
    )
    return items


def _LenRabi_make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(
    cfg: LenRabiCfg, *, context: RunContext
) -> RunRecord[LenRabiCfg, LenRabiResult]:
    """Reject an invalid length sweep before starting the core measurement."""
    sweep2array(
        cfg.length, "time", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
    )
    return RunRecord(cfg=cfg, result=LenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=LenRabiExp.AXES_SPEC,
    make_form=_LenRabi_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_LenRabi_analyze,
        options=lambda source, env: LenRabiAnalyzeOptions(),
        writeback=_LenRabi_get_writeback_items,
    ),
)
