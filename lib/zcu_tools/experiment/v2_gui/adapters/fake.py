"""FakeAdapter — no-hardware stub for testing the GUI framework."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    ParamSpec,
    SavePaths,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WritebackItem,
)

FakeResult = np.ndarray
FakeAnalyzeResult = tuple[float, Optional[Figure]]


class FakeAdapter(AbsExpAdapter[FakeResult, FakeAnalyzeResult]):
    """Minimal stub adapter — drives the full GUI flow without hardware."""

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        spec = CfgSectionSpec(
            fields={
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "sweep": SweepSpec(label="Frequency"),
                "gain": ScalarSpec(label="Gain", type=float, decimals=4),
            }
        )
        value = CfgSectionValue(
            fields={
                "reps": ScalarValue(100),
                "rounds": ScalarValue(10),
                "sweep": SweepValue(start=5.0, stop=6.0, expts=11),
                "gain": ScalarValue(0.1),
            }
        )
        return CfgSchema(spec=spec, value=value)

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
        overrides: Optional[dict] = None,  # noqa: ARG002
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
