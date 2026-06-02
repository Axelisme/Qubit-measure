"""FakeAdapter — no-hardware stub for testing the GUI framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Annotated, ClassVar, Optional, Sequence, TypeAlias, cast

from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackRequest,
)


@dataclass(frozen=True)
class FakeResult:
    data: np.ndarray
    cfg_snapshot: Optional[FakeExpCfg] = None


class FakeExpCfg(ExpCfgModel):
    reps: int = 100
    rounds: int = 10
    gain: float = 0.1
    noise_scale: float = 0.1
    sweep: object


class FakeExp(AbsExperiment[FakeResult, FakeExpCfg]):
    def run(self, cfg: FakeExpCfg) -> FakeResult:
        rng = np.random.default_rng(seed=42)
        signals = rng.normal(0.0, cfg.noise_scale, size=11)
        return FakeResult(data=signals)

    def save(self, filepath: str, result: Optional[FakeResult] = None) -> None:
        pass


FakeRunResult: TypeAlias = FakeResult


@dataclass
class FakeAnalyzeResult(AnalyzeResultBase):
    peak: float
    figure: Figure


@dataclass
class FakeAnalyzeParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)]


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
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "sweep": SweepSpec(label="Frequency"),
                "gain": ScalarSpec(label="Gain", type=float, decimals=4),
                "noise_scale": ScalarSpec(label="Noise Scale", type=float, decimals=4),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        del ctx
        return CfgSectionValue(
            fields={
                "reps": DirectValue(100),
                "rounds": DirectValue(10),
                "sweep": SweepValue(start=5.0, stop=6.0, expts=11),
                "gain": DirectValue(0.1),
                "noise_scale": DirectValue(0.1),
            }
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FakeExpCfg:
        return FakeExpCfg(
            reps=_require_int(raw_cfg, "reps"),
            rounds=_require_int(raw_cfg, "rounds"),
            gain=_require_float(raw_cfg, "gain"),
            noise_scale=_require_float(raw_cfg, "noise_scale"),
            sweep=raw_cfg["sweep"],
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> FakeRunResult:
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        return FakeExp().run(cfg)

    def get_analyze_params(
        self, result: FakeRunResult, ctx: ExpContext
    ) -> FakeAnalyzeParams:
        return FakeAnalyzeParams(threshold=0.5)

    def analyze(
        self, req: AnalyzeRequest[FakeRunResult, FakeAnalyzeParams]
    ) -> FakeAnalyzeResult:
        threshold = req.analyze_params.threshold
        data = req.run_result.data

        peak = float(np.max(np.abs(data)))
        fig = Figure()
        ax = cast(Axes, fig.subplots())
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
        return FakeAnalyzeResult(peak=peak, figure=fig)

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

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_fake"
