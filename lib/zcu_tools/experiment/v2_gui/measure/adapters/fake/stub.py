"""FakeAdapter — no-hardware stub for testing the GUI framework."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
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
from zcu_tools.gui.cfg import (
    SweepSpec,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg


@dataclass(frozen=True)
class FakeResult:
    data: NDArray[np.float64]


class FakeExpCfg(ExpCfgModel):
    reps: int = 100
    rounds: int = 10
    gain: float = 0.1
    noise_scale: float = 0.1
    sweep: SweepCfg


FakeRunResult: TypeAlias = RunRecord[FakeExpCfg, FakeResult]


@dataclass(frozen=True)
class FakeAnalyzeOptions:
    threshold: float = 0.5


@dataclass(frozen=True)
class FakeAnalysis:
    peak: float


class FakeExp:
    """Fixed seeded harness with explicit records and intentionally inert save."""

    def run(self, config: FakeExpCfg, *, context: RunContext) -> FakeResult:
        del context  # The fixed harness has no hardware or live Run presentation.
        rng = np.random.default_rng(seed=42)
        signals = rng.normal(0.0, config.noise_scale, size=11)
        return FakeResult(data=signals)

    def analyze(
        self, source: FakeRunResult, options: FakeAnalyzeOptions, *, plots: Plots
    ) -> FakeAnalysis:
        threshold = options.threshold
        data = source.result.data
        peak = float(np.max(np.abs(data)))
        _, ax = plots.subplots("fit")
        xs = np.arange(len(data))
        ax.plot(xs, data, label="signal")
        if peak > threshold:
            idx = int(np.argmax(np.abs(data)))
            ax.axvline(idx, color="red", linestyle="--", label=f"peak={peak:.3f}")
        ax.axhline(
            threshold, color="gray", linestyle=":", label=f"threshold={threshold}"
        )
        ax.set_title("FakeAdapter analysis")
        ax.legend()
        return FakeAnalysis(peak=peak)

    def save(
        self,
        source: FakeRunResult,
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None:
        """The no-hardware harness intentionally leaves data persistence inert."""
        _ = source, destination, comment, tag

    def load(self, source: Path) -> FakeRunResult:
        del source
        raise NotImplementedError(
            "The inert FakeAdapter harness has no data file format"
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
