from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

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
from zcu_tools.gui.cfg import CfgSchema, EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import custom, path_seed
from zcu_lab.v2.twotone.reset.single_tone.freq.core import FreqCfg, FreqExp, FreqResult

SingleToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


def _reset_freq_range(_ctx: SessionEnv) -> SweepValue:
    """Store the calibrated sideband frequency ± 1.5 FWHM in MHz."""
    return SweepValue(
        start=EvalValue("$qubit.ext.reset_f - 1.5 * $qubit.ext.reset_fwhm"),
        stop=EvalValue("$qubit.ext.reset_f + 1.5 * $qubit.ext.reset_fwhm"),
        expts=201,
    )


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
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset frequency sweep: drives the tested "
            "pulse-reset tone and sweeps its frequency, fitting the qubit "
            "response (Lorentzian) to find the reset sideband frequency and "
            "its linewidth. Runs on real hardware. Typically run after an "
            "init pulse excites the qubit, to locate the f01+f12 (r_f - q_f) "
            "sideband at which the reset pulse dumps the excitation."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_f and $qubit.ext.reset_fwhm for the "
            "sweep centre and half-span of 1.5 FWHM. Missing, null or invalid "
            "values fail cfg lowering. Drive defaults use the captured qubit "
            "frequency and wiring; post-delay uses $resonator.kappa."
        ),
        expects_ml=(
            "Builds the tested pulse reset inline and needs a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
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
            "if reset_f is only a rough resonator-minus-qubit estimate. The reset-pulse "
            "length is held fixed here — sweep it separately afterwards."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="reset",
                label="Tested Reset",
                shape="pulse",
                init=ModuleInit.INLINE,
                locked={"pulse_cfg.freq": 0.0},
                overrides={
                    "pulse_cfg.gain": 0.3,
                    "pulse_cfg.waveform.length": 5.0,
                    "pulse_cfg.post_delay": path_seed(
                        "$resonator.kappa",
                        expr="5.0 / (2 * 3.141592653589793 * $resonator.kappa)",
                        fallback=0.8,
                    ),
                },
            )
            .readout()
            .relax_delay(path_seed("$qubit.t1", expr="1.0 * $qubit.t1", fallback=30.5))
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=custom(
                    _reset_freq_range,
                    description="single-tone reset frequency range (201 points)",
                ),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> SingleToneFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
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
