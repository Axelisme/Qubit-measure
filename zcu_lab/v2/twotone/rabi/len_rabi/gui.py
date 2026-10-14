from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)

from .form import make_form

LenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


@dataclass
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


@dataclass
class LenRabiAnalyzeResult(AnalyzeResultBase):
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    # Rabi oscillation frequency in MHz (1/us), proposed at qubit.ext.rabi_freq.
    rabi_f: float
    rabi_f_err: float
    fit_quality: dict[str, dict[str, object]] | None = None


class LenRabiAdapter(
    BaseAdapter[
        LenRabiCfg,
        LenRabiRunResult,
        LenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = LenRabiExp.persistence_tag()
    exp_cls = LenRabiExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiRunResult:
        del req
        result = LenRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def validate_run_request(self, req: RunRequest, cfg: LenRabiCfg) -> None:
        _, soccfg = require_soc_handles(req)
        sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
        )

    def analyze(
        self,
        req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> LenRabiAnalyzeResult:
        params = req.analyze_params
        analysis = LenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(decay=params.decay, fit_phase=params.fit_phase),
            plots=plots,
        )
        return LenRabiAnalyzeResult(
            pi_len=analysis.pi_len,
            pi_len_err=analysis.pi_len_err,
            pi2_len=analysis.pi2_len,
            pi2_len_err=analysis.pi2_len_err,
            rabi_f=analysis.rabi_f,
            rabi_f_err=analysis.rabi_f_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
        )

    def get_writeback_items(
        self, req: WritebackRequest[LenRabiRunResult, LenRabiAnalyzeResult]
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
        pulse = snapshot.modules.qub_pulse
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
            spec, value = module_cfg_to_value(pulse)
            value.with_field("waveform.length", length)
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
                proposed_value=gain * result.pi_len,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_len_rabi_{time.strftime('%m%d')}"
