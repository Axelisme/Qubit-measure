from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import path_seed
from zcu_lab.v2.twotone.reset.dual_tone.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)

DualTonePowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class DualTonePowerAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualTonePowerAnalyzeResult(AnalyzeResultBase):
    gain1: float
    gain2: float


class DualTonePowerAdapter(
    BaseAdapter[
        PowerCfg,
        DualTonePowerRunResult,
        DualTonePowerAnalyzeResult,
        DualTonePowerAnalyzeParams,
    ]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Dual-tone reset gain map: a 2D sweep of the two reset-tone gains "
            "(pulse1 × pulse2 of the tested two-pulse reset) at the calibrated "
            "sideband frequencies, imaging the residual excitation to pick the "
            "gain pair that resets most completely. Runs on real hardware."
        ),
        expects_md=(
            "Requires $qubit.ext.reset_f1 and $qubit.ext.reset_f2 for the "
            "two tested-reset frequencies. Missing, null or invalid values "
            "fail cfg lowering. Other drive defaults use captured qubit "
            "frequency and wiring."
        ),
        expects_ml=(
            "Builds the tested two-pulse-reset cfg from captured defaults and "
            "needs a readout module. Optionally references an upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "Proposes the best gains to $qubit.ext.reset_gain1 and "
            "$qubit.ext.reset_gain2. With Run cfg and capture, proposes "
            "$qubit.reset.reset_120 from tested_reset, replacing only the "
            "two gains with this analysis. Tested frequencies and all other "
            "fields remain. An optional $qubit.module.reset row selects it."
        ),
        recommended=(
            "Sweep each gain across its full usable range. Analysis denoises "
            "the map before picking the optimum; wavelet smoothing is the "
            "default and Gaussian remains available for comparison. The "
            "frequencies are held at 'reset_f1' / 'reset_f2', so calibrate "
            "those first."
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
                role_id="two_pulse_reset",
                label="Tested Reset",
                shape="two_pulse",
                locked={"pulse1_cfg.gain": 0.0, "pulse2_cfg.gain": 0.0},
                overrides={
                    "pulse1_cfg.freq": path_seed("$qubit.ext.reset_f1", fallback=0.0),
                    "pulse2_cfg.freq": path_seed("$qubit.ext.reset_f2", fallback=0.0),
                },
            )
            .readout()
            .relax_delay(0.5)
            .sweep(
                "gain1",
                label="Gain 1 (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .sweep(
                "gain2",
                label="Gain 2 (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=51),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> DualTonePowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualTonePowerRunResult, DualTonePowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualTonePowerAnalyzeResult:
        params = req.analyze_params
        result = PowerExp().analyze(
            req.run_result,
            PowerAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return DualTonePowerAnalyzeResult(gain1=result.gain1, gain2=result.gain2)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualTonePowerRunResult, DualTonePowerAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose ext gains and the Run-tested reset with those fitted gains.

        None cfg or capture retains only scalars. Capture resolution and
        materialization failures propagate; tested frequencies stay unchanged.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_gain1",
                description="Dual-tone reset gain 1 (a.u.)",
                proposed_value=result.gain1,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_gain2",
                description="Dual-tone reset gain 2 (a.u.)",
                proposed_value=result.gain2,
            ),
        ]
        cfg = req.run_result.cfg
        parameters = req.ctx.parameters
        if cfg is None or parameters is None:
            return items

        root = parameters.resolve_path("$qubit.reset.reset_120")
        spec, value = module_cfg_to_value(cfg.modules.tested_reset)
        value.with_field("pulse1_cfg.gain", result.gain1)
        value.with_field("pulse2_cfg.gain", result.gain2)
        items.append(
            ModuleWriteback(
                target_name="$qubit.reset.reset_120",
                description="Reset with two pulses from 1 to 2 to 0",
                edit_schema=CfgSchema(spec=spec, value=value),
            )
        )
        items.append(
            ParameterWriteback(
                target_name="$qubit.module.reset",
                description="Select dual-tone reset for downstream experiments",
                proposed_value=root,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_dualreset_gain_{time.strftime('%m%d')}"
