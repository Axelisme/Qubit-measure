from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from typeguard import check_type

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D


class FakeExpCfg(TypedDict, total=False):
    sweep_points: int
    step_delay_s: float
    center_mhz: float
    width_mhz: float


@dataclass
class FakeSOC:
    connected: bool = True

    def get_sample_rates(self) -> Dict[str, float]:
        return {"dac": 9.8304e9, "adc": 4.9152e9}

    def ping(self) -> str:
        return "pong"


@dataclass
class FakeRunResult:
    x: list[float]
    y: list[float]
    partial: bool


class FakeExperiment(AbsExperiment[FakeRunResult, FakeExpCfg]):
    """A deterministic fake experiment for GUI interaction testing."""

    def __init__(self) -> None:
        super().__init__()
        self.last_cfg: FakeExpCfg = {}
        self.last_result: Optional[FakeRunResult] = None

    def run(
        self,
        soc: FakeSOC,
        soccfg: Any,
        cfg: dict[str, Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> FakeRunResult:
        cfg = dict(cfg)
        points = int(cfg.get("sweep_points", 101))
        delay_s = float(cfg.get("step_delay_s", 0.03))
        center = float(cfg.get("center_mhz", 6800.0))
        width = float(cfg.get("width_mhz", 5.0))

        freqs = np.linspace(center - width, center + width, points, dtype=float)

        def measure_fn(
            ctx: TaskState[Any, Any],
            update_hook: Callable[[int, Any], None],
        ) -> list[np.ndarray]:
            values: list[float] = []
            rng = np.random.default_rng(42)
            for idx, freq in enumerate(freqs):
                if should_cancel is not None and should_cancel():
                    break
                y = np.exp(-((freq - center) ** 2) / (2 * (0.25 * width) ** 2))
                y += 0.03 * np.sin((freq - center) * 2.0)
                y += float(rng.normal(scale=0.01))
                values.append(float(y))
                if update_hook is not None:
                    update_hook(idx + 1, values)
            return [np.asarray(values, dtype=float)]

        signals = run_task(
            task=Task(
                measure_fn=measure_fn,
                raw2signal_fn=lambda raw: np.asarray(raw[0], dtype=np.complex128),
                result_shape=(points,),
                pbar_n=points,
            ),
            init_cfg=cfg,
        )

        signals = np.asarray(signals, dtype=np.complex128)
        result = FakeRunResult(
            x=[float(v) for v in freqs[: len(signals)]],
            y=[float(np.real(v)) for v in signals],
            partial=len(signals) < points,
        )
        self.last_cfg = check_type(deepcopy(cfg), FakeExpCfg)
        self.last_result = result
        return result

    def analyze(self, result: Optional[FakeRunResult] = None) -> Dict[str, Any]:
        if result is None:
            result = self.last_result
        if result is None or not result.x:
            raise RuntimeError("No run result available to analyze")

        x = np.asarray(result.x, dtype=float)
        y = np.asarray(result.y, dtype=float)
        peak_idx = int(np.argmax(y))
        return {
            "peak_freq_mhz": float(x[peak_idx]),
            "peak_amp": float(y[peak_idx]),
            "points": int(len(x)),
            "partial": bool(result.partial),
        }

    def save(
        self,
        filepath: str,
        result: Optional[FakeRunResult] = None,
        comment: Optional[str] = None,
        tag: str = "v2_gui/mock",
        **kwargs: Any,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.last_cfg,
            "result": {"x": result.x, "y": result.y, "partial": result.partial},
            "comment": comment,
            "tag": tag,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, filepath: str, **kwargs: Any) -> FakeRunResult:
        raw = json.loads(Path(filepath).read_text(encoding="utf-8"))
        result = FakeRunResult(
            x=[float(v) for v in raw["result"]["x"]],
            y=[float(v) for v in raw["result"]["y"]],
            partial=bool(raw["result"].get("partial", False)),
        )
        self.last_cfg = check_type(deepcopy(raw.get("cfg", {})), FakeExpCfg)
        self.last_result = result
        return result

    def save_run(
        self, filepath: Path, cfg: Dict[str, Any], result: FakeRunResult
    ) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": cfg,
            "result": {"x": result.x, "y": result.y, "partial": result.partial},
        }
        filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return filepath

    def save_analysis_figure(self, filepath: Path, result: FakeRunResult) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(result.x, result.y, lw=1.4)
        ax.set_title("Fake Analysis")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        fig.savefig(filepath, dpi=120)
        plt.close(fig)
        return filepath


@dataclass
class FakeDevice:
    name: str
    info: Dict[str, Any]

    def set_field(self, field: str, value: Any) -> None:
        if field not in self.info:
            raise KeyError(f"Unknown field: {field}")
        if field == "power_dBm" and not (-120.0 <= float(value) <= 25.0):
            raise ValueError("power_dBm out of range")
        if field == "freq_Hz" and not (1e6 <= float(value) <= 20e9):
            raise ValueError("freq_Hz out of range")
        self.info[field] = value

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
