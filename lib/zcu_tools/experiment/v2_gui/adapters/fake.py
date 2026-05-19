"""FakeAdapter — no-hardware stub for testing the GUI framework."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSection,
    ExpContext,
    ParamSpec,
    SavePaths,
    ScalarField,
    SweepField,
    WritebackItem,
)

FakeResult = np.ndarray
FakeAnalyzeResult = tuple[float, Optional[Figure]]


class FakeAdapter(AbsExpAdapter[FakeResult, FakeAnalyzeResult]):
    """Minimal stub adapter — drives the full GUI flow without hardware."""

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        root = CfgSection(
            fields={
                "reps": ScalarField(value=100, label="Reps", type=int),
                "rounds": ScalarField(value=10, label="Rounds", type=int),
                "sweep": SweepField(start=5.0, stop=6.0, expts=11, label="Frequency"),
                "gain": ScalarField(value=0.1, label="Gain", type=float),
            }
        )
        return CfgSchema(root=root)

    def get_run_params(self) -> dict[str, ParamSpec]:
        return {
            "noise_scale": ParamSpec(
                label="Noise Scale", default=0.1, type=float, choices=None
            ),
        }

    def run(
        self,
        ctx: ExpContext,  # noqa: ARG002
        schema: CfgSchema,  # noqa: ARG002
        **user_params: Any,
    ) -> FakeResult:
        noise_scale = float(user_params.get("noise_scale", 0.1))
        rng = np.random.default_rng(seed=42)
        return rng.normal(0.0, noise_scale, size=11)

    def get_analyze_params(self) -> dict[str, ParamSpec]:
        return {
            "threshold": ParamSpec(
                label="Threshold", default=0.5, type=float, choices=None
            ),
        }

    def analyze(
        self,
        result: FakeResult,
        ctx: ExpContext,  # noqa: ARG002
        **user_params: Any,
    ) -> FakeAnalyzeResult:
        threshold = float(user_params.get("threshold", 0.5))
        peak = float(np.max(np.abs(result)))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
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
        plt.close(fig)
        return (peak, fig)

    def get_figure(self, analyze_result: FakeAnalyzeResult) -> Optional[Figure]:
        return analyze_result[1]

    def get_writeback_spec(
        self,
        analyze_result: FakeAnalyzeResult,
        ctx: ExpContext,  # noqa: ARG002
    ) -> list[WritebackItem]:
        peak, _ = analyze_result
        return [
            WritebackItem(
                key="fake_peak",
                target="md",
                current_value=0.0,
                new_value=peak,
                description="Fake peak value from FakeAdapter analysis",
            )
        ]

    def apply_writeback(
        self,
        ctx: ExpContext,  # noqa: ARG002
        analyze_result: FakeAnalyzeResult,  # noqa: ARG002
        selected_keys: list[str],  # noqa: ARG002
    ) -> None:
        pass  # nothing to persist in fake mode

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:  # noqa: ARG002
        return SavePaths(data_path="/tmp/fake_data", image_path="/tmp/fake_image.png")

    def save(
        self,
        data_path: str,  # noqa: ARG002
        result: FakeResult,  # noqa: ARG002
        ctx: ExpContext,  # noqa: ARG002
    ) -> None:
        pass  # nothing to save in fake mode
