"""stub experiment GUI attachment."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import SweepSpec, SweepValue
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2.fake.stub.core import (
    FakeAnalyzeOptions,
    FakeExp,
    FakeExpCfg,
    FakeRunResult,
)


@dataclass
class FakeAnalyzeResult(AnalyzeResultBase):
    peak: float


@dataclass
class FakeAnalyzeParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)] = 0.5


def _require_int(raw_cfg: dict[str, object], key: str) -> int:
    value = raw_cfg.get(key)
    if not isinstance(value, int):
        raise RuntimeError(
            f"FakeAdapter config field {key!r} must be int, got {type(value)}"
        )
    return value


def _require_float(raw_cfg: dict[str, object], key: str) -> float:
    value = raw_cfg.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"FakeAdapter config field {key!r} must be float, got {type(value)}"
        )
    return float(value)


class FakeAdapter(
    BaseAdapter[FakeExpCfg, FakeRunResult, FakeAnalyzeResult, FakeAnalyzeParams]
):
    """Minimal stub adapter — drives the full GUI flow without hardware."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )
    exp_cls = FakeExp

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reps(100)
            .rounds(10)
            .field(
                "sweep",
                spec=SweepSpec(label="Frequency"),
                default=SweepValue(start=5.0, stop=6.0, expts=11),
            )
            .float("gain", label="Gain", default=0.1, decimals=4)
            .float("noise_scale", label="Noise Scale", default=0.1, decimals=4)
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FakeExpCfg:
        return FakeExpCfg(
            reps=_require_int(raw_cfg, "reps"),
            rounds=_require_int(raw_cfg, "rounds"),
            gain=_require_float(raw_cfg, "gain"),
            noise_scale=_require_float(raw_cfg, "noise_scale"),
            sweep=SweepCfg.model_validate(raw_cfg["sweep"]),
            dev=deepcopy(req.device_snapshot),
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> FakeRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FakeExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[FakeRunResult, FakeAnalyzeParams], *, plots: Plots
    ) -> FakeAnalyzeResult:
        analysis = FakeExp().analyze(
            req.run_result,
            FakeAnalyzeOptions(threshold=req.analyze_params.threshold),
            plots=plots,
        )
        return FakeAnalyzeResult(peak=analysis.peak)

    def get_writeback_items(
        self, req: WritebackRequest[FakeRunResult, FakeAnalyzeResult]
    ) -> Sequence[MetaDictWriteback]:
        return [
            MetaDictWriteback(
                target_name="fake_peak",
                description="Fake peak value from FakeAdapter analysis",
                proposed_value=req.analyze_result.peak,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_fake"
