from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.single_tone.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.twotone.reset.single_tone.freq.form import make_form

SingleToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class SingleToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float


class SingleToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        SingleToneFreqRunResult,
        SingleToneFreqAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset frequency sweep: drives the tested "
            "pulse-reset tone and sweeps its frequency, fitting the qubit "
            "response (Lorentzian) to find the reset sideband frequency and "
            "its linewidth. Runs on real hardware. Typically run after an "
            "init pulse excites the qubit, to locate the f01+f12 ($resonator.freq - $qubit.freq) "
            "sideband at which the reset pulse dumps the excitation."
        ),
        expects_md=(
            "Seeds 201 frequency points at $qubit.ext.reset_f +/- 1.5 * "
            "$qubit.ext.reset_fwhm. The tested pulse has gain 0.3, length 5.0 us, and "
            "post-delay 5.0 / (2 * pi * $resonator.kappa) us. Relax delay reads "
            "$qubit.t1. Open and Reset evaluate starting expressions once into direct "
            "input using captured sources. Inline qubit pulses read $qubit.freq and "
            "$qubit.wiring.L01.ch. Inline readout reads $resonator.freq, "
            "$resonator.wiring.ch, $resonator.wiring.ro_ch and $resonator.ext.time_of_fly "
            "+ 0.05 us for trigger timing. Missing or invalid required sources leave the "
            "affected fields empty and Invalid; supply calibration or direct values "
            "before Run. No numeric calibration fallback is supplied."
        ),
        expects_ml=(
            "Tested reset starts inline from the reset template. Optional init pulse uses "
            "$qubit.module.x180 without fallback; an unset/dangling slot or unbound alias "
            "leaves it disabled and Valid. $qubit.module.readout selects a captured "
            "library link, otherwise a fresh inline readout template with a const "
            "waveform. Its required parameter sources must still be present. Optional "
            "$qubit.module.reset has no fallback: an unset/dangling slot or unbound alias "
            "leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes fitted frequency and FWHM to $qubit.ext.reset_f and "
            "$qubit.ext.reset_fwhm. With Run cfg and capture, proposes "
            "$qubit.reset.reset_10 from tested_reset, replacing only its "
            "frequency with this analysis. All other tested fields remain. "
            "An optional $qubit.module.reset row selects the concrete root."
        ),
        recommended=(
            "A sweep of ~201 points spanning a couple of linewidths around "
            "the expected sideband captures the dip cleanly; widen the span "
            "if $qubit.ext.reset_f is only a rough resonator-minus-qubit estimate. The reset-pulse "
            "length is held fixed here — sweep it separately afterwards."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> SingleToneFreqRunResult:
        del req
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SingleToneFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> SingleToneFreqAnalyzeResult:
        result = FreqExp().analyze(req.run_result, None, plots=plots)
        return SingleToneFreqAnalyzeResult(freq=result.freq, fwhm=result.fwhm)

    def get_writeback_items(
        self,
        req: WritebackRequest[SingleToneFreqRunResult, SingleToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose ext scalars and the Run-tested reset with fitted frequency.

        None cfg or capture retains only scalars. Capture resolution and
        materialization failures propagate; no calibration is reread.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_f",
                description="Sideband reset frequency (MHz)",
                proposed_value=result.freq,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_fwhm",
                description="Reset sideband linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]
        cfg = req.run_result.cfg
        parameters = req.ctx.parameters
        if cfg is None or parameters is None:
            return items

        root = parameters.resolve_path("$qubit.reset.reset_10")
        spec, value = module_cfg_to_value(cfg.modules.tested_reset)
        value.with_field("pulse_cfg.freq", result.freq)
        items.append(
            ModuleWriteback(
                target_name="$qubit.reset.reset_10",
                description="Reset with one pulse from 1 to 0",
                edit_schema=CfgSchema(spec=spec, value=value),
            )
        )
        items.append(
            ParameterWriteback(
                target_name="$qubit.module.reset",
                description="Select single-tone reset for downstream experiments",
                proposed_value=root,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sidereset_freq_{time.strftime('%m%d')}"
