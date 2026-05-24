"""FakeAdapter — no-hardware stub for testing the GUI framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zcu_tools.experiment.base import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeParam,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WritebackRequest,
)

FakeResult = np.ndarray


class FakeExpCfg(ExpCfgModel):
    reps: int = 100
    rounds: int = 10
    gain: float = 0.1
    noise_scale: float = 0.1
    sweep: object


class FakeExperiment(AbsExperiment[FakeResult, FakeExpCfg]):
    def run(self, cfg: FakeExpCfg) -> FakeResult:
        rng = np.random.default_rng(seed=42)
        return rng.normal(0.0, cfg.noise_scale, size=11)


@dataclass
class FakeAnalyzeResult:
    peak: float
    figure: Figure


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


class FakeAdapter(AbsExpAdapter[FakeResult, FakeAnalyzeResult]):
    """Minimal stub adapter — drives the full GUI flow without hardware."""

    exp_cls = FakeExperiment

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        spec = CfgSectionSpec(
            fields={
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "sweep": SweepSpec(label="Frequency"),
                "gain": ScalarSpec(label="Gain", type=float, decimals=4),
                "noise_scale": ScalarSpec(label="Noise Scale", type=float, decimals=4),
            }
        )
        value = CfgSectionValue(
            fields={
                "reps": DirectValue(100),
                "rounds": DirectValue(10),
                "sweep": SweepValue(start=5.0, stop=6.0, expts=11),
                "gain": DirectValue(0.1),
                "noise_scale": DirectValue(0.1),
            }
        )
        return CfgSchema(spec=spec, value=value)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FakeExpCfg:  # noqa: ARG002
        return FakeExpCfg(
            reps=_require_int(raw_cfg, "reps"),
            rounds=_require_int(raw_cfg, "rounds"),
            gain=_require_float(raw_cfg, "gain"),
            noise_scale=_require_float(raw_cfg, "noise_scale"),
            sweep=raw_cfg["sweep"],
        )

    def get_analyze_params(
        self,
        result: FakeResult,  # noqa: ARG002
        ctx: ExpContext,  # noqa: ARG002
    ) -> list[AnalyzeParam]:
        return [
            AnalyzeParam(
                key="threshold",
                label="Threshold",
                type=float,
                default=0.5,
            )
        ]

    def analyze(
        self,
        req: AnalyzeRequest[FakeResult],
    ) -> FakeAnalyzeResult:
        threshold_value = req.analyze_params.get("threshold")
        if not isinstance(threshold_value, (int, float)) or isinstance(
            threshold_value, bool
        ):
            raise RuntimeError("Analyze param 'threshold' must be float")
        threshold = float(threshold_value)
        result = req.run_result
        if not isinstance(result, np.ndarray):
            raise RuntimeError("Analyze request run_result must be numpy.ndarray")
        peak = float(np.max(np.abs(result)))
        fig = Figure()
        ax = cast(Axes, fig.subplots())
        xs = np.arange(len(result))
        ax.plot(xs, result, label="signal")
        if peak > threshold:
            idx = int(np.argmax(np.abs(result)))
            ax.axvline(idx, color="red", linestyle="--", label=f"peak={peak:.3f}")
        ax.axhline(
            threshold, color="gray", linestyle=":", label=f"threshold={threshold}"
        )
        ax.set_title("FakeAdapter analysis")
        ax.legend()
        return FakeAnalyzeResult(peak=peak, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[FakeResult, FakeAnalyzeResult],
    ) -> Sequence[MetaDictWriteback]:
        return [
            MetaDictWriteback(
                key="fake_peak",
                description="Fake peak value from FakeAdapter analysis",
                current_value=0.0,
                md_key="fake_peak",
                proposed_value=req.analyze_result.peak,
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:  # noqa: ARG002
        return f"{ctx.res_name}_fake"

    def save(
        self,
        req: SaveDataRequest[FakeResult],  # noqa: ARG002
    ) -> None:
        pass  # nothing to save in fake mode
