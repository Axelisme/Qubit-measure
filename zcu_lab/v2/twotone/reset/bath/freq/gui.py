from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import Seed, custom, scaled_path_seed
from zcu_lab.v2.twotone.reset.bath.freq.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)

BathFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]

# The pi/2 readout tomography pulse sits at a fixed phase offset; the domain
# adds a 4-point QickSweep1D("phase", 0, 270) on top of pi2_cfg.phase, so the
# form value is the *base* offset (notebook: phase=90), not a swept axis.
_PI2_PHASE_OFFSET_DEG: float = 90.0


def _cavity_freq_range(expts: int) -> Seed[SweepValue]:
    """Store cavity frequency minus 0.8–1.2 times calibrated Rabi frequency."""

    def resolve(_ctx: SessionEnv) -> SweepValue:
        return SweepValue(
            start=EvalValue("$resonator.freq - 1.2 * $qubit.ext.rabi_freq"),
            stop=EvalValue("$resonator.freq - 0.8 * $qubit.ext.rabi_freq"),
            expts=expts,
        )

    return custom(resolve, description="bath cavity frequency range")


@dataclass
class BathFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class BathFreqGainAnalyzeResult(AnalyzeResultBase):
    gain: float
    freq: float


class BathFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        BathFreqGainRunResult,
        BathFreqGainAnalyzeResult,
        BathFreqGainAnalyzeParams,
    ]
):
    exp_cls = FreqGainExp
    ExpCfg_cls: ClassVar[Any] = FreqGainCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Bath-reset cavity frequency–gain map: a 2D sweep of the bath "
            "reset's cavity-tone frequency × gain, with an internal 4-point "
            "pi/2 tomography phase axis the experiment drives itself, imaging "
            "the residual coherence to locate the (freq, gain) pair that best "
            "dumps the excitation into the cavity bath. Runs on real hardware."
        ),
        expects_md=(
            "Requires $resonator.freq and $qubit.ext.rabi_freq for the "
            "cavity sweep from resonator frequency minus 1.2 Rabi frequency "
            "to minus 0.8 Rabi frequency. Other drive defaults use captured "
            "qubit/resonator frequency and wiring. Missing, null or invalid "
            "seed values fail cfg lowering."
        ),
        expects_ml=(
            "Builds the tested bath-reset cfg with cavity, qubit and pi/2 "
            "tones from captured defaults and needs a readout module. "
            "Optionally references an upstream reset and an init pulse "
            "(a library pi pulse when present), both disabled when no "
            "library entry exists."
        ),
        typical_writeback=(
            "Proposes the best cavity gain and frequency to "
            "$qubit.ext.reset_bath_gain and $qubit.ext.reset_bath_freq only. "
            "BathPhase proposes the tested ground/excited reset modules; "
            "length analysis has no automatic writeback."
        ),
        recommended=(
            "Analysis denoises the 2D map before picking the peak; wavelet "
            "smoothing is the default and Gaussian remains available for "
            "comparison. Save writes one 3D HDF5 file with the tomography phase "
            "stored as an internal sweep axis."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset("reset", optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="bath_reset",
                label="Tested Reset",
                shape="bath",
                locked={
                    "cavity_tone_cfg.freq": 0.0,
                    "cavity_tone_cfg.gain": 0.0,
                    "pi2_cfg.phase": _PI2_PHASE_OFFSET_DEG,
                },
            )
            .readout()
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5),
            )
            .sweep(
                "freq",
                label="Cavity freq (MHz)",
                default=_cavity_freq_range(51),
            )
            .sweep(
                "gain",
                label="Cavity gain (a.u.)",
                default=SweepValue(start=0.4, stop=1.0, expts=51),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> BathFreqGainRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FreqGainExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathFreqGainRunResult, BathFreqGainAnalyzeParams],
        *,
        plots: Plots,
    ) -> BathFreqGainAnalyzeResult:
        params = req.analyze_params
        analysis = FreqGainExp().analyze(
            req.run_result,
            FreqGainAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return BathFreqGainAnalyzeResult(gain=analysis.gain, freq=analysis.freq)

    def get_writeback_items(
        self,
        req: WritebackRequest[BathFreqGainRunResult, BathFreqGainAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's cavity gain then frequency to qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_gain",
                description="Bath-reset cavity gain (a.u.)",
                proposed_value=result.gain,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_freq",
                description="Bath-reset cavity frequency (MHz)",
                proposed_value=result.freq,
            ),
        ]
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_bathreset_freqgain_{time.strftime('%m%d')}"
