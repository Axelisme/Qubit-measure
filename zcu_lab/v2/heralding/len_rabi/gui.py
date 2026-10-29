"""GUI wiring and ordinary length-Rabi writeback targets for heralded data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalysis,
    LenRabiAnalyzeOptions,
)

from .core import HeraldingLenRabiCfg, HeraldingLenRabiExp, HeraldingLenRabiResult
from .form import make_form

LenRabiRunResult: TypeAlias = RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]


def analyze(
    req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeOptions], *, plots: Plots
) -> LenRabiAnalysis:
    """Fit finite selected IQ through the existing length-Rabi owner."""
    return HeraldingLenRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )


def writeback(
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
            description="Heralded Rabi oscillation frequency (MHz)",
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
        (
            "$qubit.pulses.pi_amp",
            "$qubit.module.x180",
            result.pi_len,
            "Heralded Pi pulse",
        ),
        (
            "$qubit.pulses.pi2_amp",
            "$qubit.module.x90",
            result.pi2_len,
            "Heralded Pi/2 pulse",
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
            description="Heralded Pi pulse gain times length (a.u. * us)",
            proposed_value=gain * result.pi_len,
        )
    )
    return items


def _run(
    cfg: HeraldingLenRabiCfg, *, context: RunContext
) -> RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult]:
    """Keep the unswept cfg and its readout classifier with selected results."""
    return RunRecord(cfg=cfg, result=HeraldingLenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=HeraldingLenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=analyze,
        options=lambda source, env: LenRabiAnalyzeOptions(),
        writeback=writeback,
    ),
)
