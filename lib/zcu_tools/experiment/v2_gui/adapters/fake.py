from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from zcu_tools.experiment.v2.fake import FakeExp, FakeResult, fake_signal2real
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .base import ConfigFieldSchema, ExperimentAdapterBase


class FakeExperimentAdapter(ExperimentAdapterBase[dict[str, Any]]):
    """Adapter for FakeExp used by current mock GUI."""

    def __init__(self, exp: FakeExp, *, scope_name: str = "v2_gui_run") -> None:
        super().__init__(exp, scope_name=scope_name)
        self.exp: FakeExp = exp

    def _run_exp(self) -> None:
        self.exp.run()

    def get_config_schema(self) -> list[ConfigFieldSchema]:
        return [
            ConfigFieldSchema(
                key="sweep_points",
                label="Sweep Points",
                field_type="int",
                default=101,
                minimum=3,
            ),
            ConfigFieldSchema(
                key="step_delay_s",
                label="Step Delay (s)",
                field_type="float",
                default=0.02,
                minimum=0.0,
            ),
            ConfigFieldSchema(
                key="center_mhz",
                label="Center (MHz)",
                field_type="float",
                default=6812.3,
            ),
            ConfigFieldSchema(
                key="width_mhz",
                label="Width (MHz)",
                field_type="float",
                default=10.0,
                minimum=0.01,
            ),
        ]

    def build_default_config(self) -> dict[str, Any]:
        return super().build_default_config()

    def analyze(self) -> dict[str, Any]:
        source = self._resolve_source()
        freqs, signals = source
        y = fake_signal2real(signals)
        if len(freqs) == 0:
            raise RuntimeError("No run result available to analyze")
        peak_idx = int(np.argmax(y))
        return {
            "peak_freq_mhz": float(freqs[peak_idx]),
            "peak_amp": float(y[peak_idx]),
            "points": int(len(freqs)),
        }

    def save_run(self, filepath: Path, cfg: dict[str, Any]) -> Path:
        source = self._resolve_source()
        x, y = self._to_payload(source)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": cfg,
            "result": {"x": x, "y": y},
        }
        filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return filepath

    def save_analysis_figure(self, filepath: Path) -> Path:
        source = self._resolve_source()
        fig = self.exp.analyze(source)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=120)
        return filepath

    def apply_analysis_to_context(
        self,
        analysis: dict[str, Any],
        meta_dict: MetaDict,
        module_library: ModuleLibrary,
    ) -> tuple[Path, Path]:
        peak_raw = analysis.get("peak_freq_mhz")
        if peak_raw is None:
            raise RuntimeError("Analysis result missing 'peak_freq_mhz'")
        peak = float(peak_raw)
        setattr(meta_dict, "r_f", peak)
        meta_dict.sync()

        module_library.sync()
        if "readout_rf" in module_library.modules:
            module_library.update_module("readout_rf", {"freq": peak})
        module_library.sync()
        if meta_dict._path is None or module_library._path is None:
            raise RuntimeError("MetaDict or ModuleLibrary path is not configured")
        return meta_dict._path, module_library._path

    def _resolve_source(self) -> FakeResult:
        if self.exp.last_result is None:
            raise RuntimeError("No run result available")
        return self.exp.last_result

    def _to_payload(self, result: FakeResult) -> tuple[list[float], list[float]]:
        freqs, signals = result
        y = fake_signal2real(signals)
        return [float(v) for v in freqs], [float(v) for v in y]
