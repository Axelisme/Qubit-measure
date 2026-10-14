from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
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
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import Seed, custom, path_seed
from zcu_tools.gui.cfg import (
    EvalValue,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.reset.dual_tone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)

DualToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


def _reset_freq_axis(
    parameter: Literal["reset_f1", "reset_f2"],
) -> Seed[SweepValue]:
    """Store a 201-point ±50 MHz window around a calibrated dual-tone center."""
    path = f"$qubit.ext.{parameter}"

    def resolve(_ctx: SessionEnv) -> SweepValue:
        return SweepValue(
            start=EvalValue(f"{path} - 50.0"),
            stop=EvalValue(f"{path} + 50.0"),
            expts=201,
        )

    return custom(resolve, description=f"{path} reset frequency axis (201 points)")


@dataclass
class DualToneFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualToneFreqAnalyzeResult(AnalyzeResultBase):
    freq1: float
    freq2: float


class DualToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        DualToneFreqRunResult,
        DualToneFreqAnalyzeResult,
        DualToneFreqAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="two_pulse_reset",
                label="Tested Reset",
                shape="two_pulse",
                locked={"pulse1_cfg.freq": 0.0, "pulse2_cfg.freq": 0.0},
                overrides={
                    "pulse1_cfg.gain": path_seed(
                        "$qubit.ext.reset_gain1", fallback=1.0
                    ),
                    "pulse2_cfg.gain": path_seed(
                        "$qubit.ext.reset_gain2", fallback=1.0
                    ),
                },
            )
            .readout()
            .relax_delay(0.5)
            .sweep(
                "freq1",
                label="Freq 1 (MHz)",
                default=_reset_freq_axis("reset_f1"),
            )
            .sweep(
                "freq2",
                label="Freq 2 (MHz)",
                default=_reset_freq_axis("reset_f2"),
            )
            .reps(100)
            .rounds(1000)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> FreqCfg:
        cfg = super().build_exp_cfg(raw_cfg, req)
        return cfg.with_updates(method="hard")

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> DualToneFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneFreqRunResult, DualToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneFreqAnalyzeResult:
        params = req.analyze_params
        result = FreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return DualToneFreqAnalyzeResult(freq1=result.freq1, freq2=result.freq2)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualToneFreqRunResult, DualToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's two frequencies to captured qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_f1",
                description="Dual-tone reset frequency 1 (MHz)",
                proposed_value=result.freq1,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_f2",
                description="Dual-tone reset frequency 2 (MHz)",
                proposed_value=result.freq2,
            ),
        ]
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_freq_{time.strftime('%m%d')}"
