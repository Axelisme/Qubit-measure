from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, TypedDict

from .fake_backend import FakeRunResult
from .state import BufferDescriptor, BufferKind, GroupDescriptor


class RunRequest(TypedDict, total=False):
    sweep_points: int
    step_delay_s: float
    center_mhz: float
    width_mhz: float


class AnalyzeResult(TypedDict, total=False):
    peak_freq_mhz: float
    peak_amp: float
    points: int
    partial: bool
    figure_path: str


class ExperimentPort(Protocol):
    def run(
        self,
        soc: Any,
        soccfg: Any,
        cfg: Dict[str, Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> FakeRunResult: ...

    def analyze(self, result: Optional[FakeRunResult] = None) -> Dict[str, Any]: ...

    def save_run(
        self, filepath: Path, cfg: Dict[str, Any], result: FakeRunResult
    ) -> Path: ...

    def save_analysis_figure(self, filepath: Path, result: FakeRunResult) -> Path: ...


@dataclass
class ExperimentGroup:
    group_id: str
    title: str
    experiment: ExperimentPort
    soc: Any
    soccfg: Any
    exp_cfg: Dict[str, Any] = field(default_factory=dict)
    last_result: Optional[FakeRunResult] = None
    last_analysis: Optional[AnalyzeResult] = None

    @classmethod
    def create(
        cls,
        group_id: str,
        title: str,
        experiment: ExperimentPort,
        soc: Any,
        soccfg: Any,
        default_cfg: Optional[RunRequest] = None,
    ) -> "ExperimentGroup":
        cfg: Dict[str, Any] = {
            "sweep_points": 101,
            "step_delay_s": 0.02,
            "center_mhz": 6812.3,
            "width_mhz": 10.0,
        }
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
                payload={"cfg": dict(self.exp_cfg), "log": []},
            ),
            BufferDescriptor(
                buffer_id=self.analyze_buffer_id,
                group_id=self.group_id,
                title="Analyze",
                kind=BufferKind.ANALYZE,
                payload={"analysis": {}},
            ),
            BufferDescriptor(
                buffer_id=self.comment_buffer_id,
                group_id=self.group_id,
                title="Comment",
                kind=BufferKind.COMMENT,
                payload={"text": ""},
            ),
            BufferDescriptor(
                buffer_id=self.artifact_buffer_id,
                group_id=self.group_id,
                title="Artifacts",
                kind=BufferKind.FILE_TEXT,
                payload={"path": ""},
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

    @property
    def comment_buffer_id(self) -> str:
        return f"{self.group_id}:comment"

    @property
    def artifact_buffer_id(self) -> str:
        return f"{self.group_id}:artifact"

    def run(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> FakeRunResult:
        result = self.experiment.run(
            self.soc,
            self.soccfg,
            dict(self.exp_cfg),
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        self.last_result = result
        return result

    def analyze(self) -> AnalyzeResult:
        analysis = self.experiment.analyze(self.last_result)
        parsed: AnalyzeResult = {
            "peak_freq_mhz": float(analysis["peak_freq_mhz"]),
            "peak_amp": float(analysis["peak_amp"]),
            "points": int(analysis["points"]),
            "partial": bool(analysis["partial"]),
        }
        self.last_analysis = parsed
        return parsed

    def save_run(self, path: Path) -> Path:
        if self.last_result is None:
            raise RuntimeError("No run result to save")
        return self.experiment.save_run(path, self.exp_cfg, self.last_result)

    def save_analysis_figure(self, path: Path) -> Path:
        if self.last_result is None:
            raise RuntimeError("No run result for analysis figure")
        saved = self.experiment.save_analysis_figure(path, self.last_result)
        if self.last_analysis is not None:
            self.last_analysis["figure_path"] = str(saved)
        return saved
