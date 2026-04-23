from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

from .adapters.base import ExperimentAdapterBase
from .state import BufferDescriptor, BufferKind, GroupDescriptor

class RunRequest(TypedDict, total=False):
    sweep_points: int
    step_delay_s: float
    center_mhz: float
    width_mhz: float


@dataclass
class ExperimentGroup:
    group_id: str
    title: str
    experiment: ExperimentAdapterBase[Dict[str, Any]]
    soc: Any
    soccfg: Any
    exp_cfg: Dict[str, Any] = field(default_factory=dict)
    last_analysis: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        group_id: str,
        title: str,
        experiment: ExperimentAdapterBase[Dict[str, Any]],
        soc: Any,
        soccfg: Any,
        default_cfg: Optional[RunRequest] = None,
    ) -> "ExperimentGroup":
        cfg: Dict[str, Any] = dict(experiment.build_default_config())
        if default_cfg:
            cfg.update(default_cfg)
        return cls(
            group_id=group_id,
            title=title,
            experiment=experiment,
            soc=soc,
            soccfg=soccfg,
            exp_cfg=cfg,
        )

    def to_descriptors(self) -> tuple[GroupDescriptor, list[BufferDescriptor]]:
        group = GroupDescriptor(group_id=self.group_id, title=self.title)
        buffers = [
            BufferDescriptor(
                buffer_id=self.run_buffer_id,
                group_id=self.group_id,
                title="Run",
                kind=BufferKind.RUN,
                payload={"cfg": dict(self.exp_cfg), "log": [], "comment": ""},
            ),
            BufferDescriptor(
                buffer_id=self.analyze_buffer_id,
                group_id=self.group_id,
                title="Analyze",
                kind=BufferKind.ANALYZE,
                payload={"analysis": {}},
            ),
        ]
        for buffer in buffers:
            group.add_buffer(buffer.buffer_id)
        return group, buffers

    @property
    def run_buffer_id(self) -> str:
        return f"{self.group_id}:run"

    @property
    def analyze_buffer_id(self) -> str:
        return f"{self.group_id}:analyze"

    def run(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.experiment.run(
            self.soc,
            self.soccfg,
            dict(self.exp_cfg),
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def analyze(self) -> Dict[str, Any]:
        analysis = dict(self.experiment.analyze())
        self.last_analysis = analysis
        return analysis

    def save_run(self, path: Path) -> Path:
        return self.experiment.save_run(path, self.exp_cfg)

    def save_analysis_figure(self, path: Path) -> Path:
        saved = self.experiment.save_analysis_figure(path)
        if self.last_analysis is not None:
            self.last_analysis["figure_path"] = str(saved)
        return saved
