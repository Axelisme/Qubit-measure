"""Session-owned completion of one analysis operation, without replay or recovery."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.session import GuiConnection, MeasureMcpSession

AnalysisStage = Literal["primary", "post"]
ExecutionStatus = Literal["running", "interactive", "finished", "failed", "cancelled"]
ExecutionPhase = Literal["operation", "result_read", "image_save", "figure_read", "terminal"]
SaveStatus = Literal["not_started", "not_available", "saving", "saved", "incomplete", "unknown"]


@dataclass(frozen=True)
class AnalysisResult:
    summary: Any
    params: dict[str, Any]
    operation_state: dict[str, Any]


@dataclass(frozen=True)
class SavedImage:
    figure_name: str
    image_path: str


@dataclass(frozen=True)
class ExecutionError:
    phase: ExecutionPhase
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class ExecutionSnapshot:
    execution: str
    tab: str
    stage: AnalysisStage
    op: int
    status: ExecutionStatus = "running"
    phase: ExecutionPhase = "operation"
    cancel_requested: bool = False
    operation_outcome: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    invalidated: list[str] | None = None
    result: AnalysisResult | None = None
    interaction: dict[str, Any] | None = None
    save_status: SaveStatus = "not_started"
    saved_images: list[SavedImage] = field(default_factory=list)
    remaining_images: list[str] | None = None
    unconfirmed_image: str | None = None
    figure: str | None = None
    error: ExecutionError | None = None


class AnalysisExecution:
    """One fixed GUI binding and detached observations of its completion."""

    def __init__(self, snapshot: ExecutionSnapshot) -> None:
        self._snapshot = snapshot
        self._images: tuple[PngImage, ...] = ()

    def snapshot(self) -> ExecutionSnapshot:
        return deepcopy(self._snapshot)

    def wait(self, timeout: float) -> ToolReply:
        """Wait locally for completion or an interactive handoff, never cancel."""
        return ToolReply(asdict(self.snapshot()), self._images)


class AnalysisExecutions:
    """Session lifetime owner; each opaque operation gets one completion owner."""

    def __init__(self, session: MeasureMcpSession) -> None:
        self._session = session
        self._next_id = 1

    def start(
        self,
        connection: GuiConnection,
        tab: str,
        stage: AnalysisStage,
        started: dict[str, Any],
    ) -> AnalysisExecution:
        """Retain the delivered start receipt, even when close wins admission."""
        execution = AnalysisExecution(
            ExecutionSnapshot(
                execution=f"analysis-{self._next_id}",
                tab=tab,
                stage=stage,
                op=started["handle"],
                params=deepcopy(started["params"]),
            )
        )
        self._next_id += 1
        return execution

    def stop_admission(self) -> None:
        """Permanently reject new workers and wake existing ones."""

    def join(self) -> None:
        """Join admitted workers after transport disconnect, before PNG cleanup."""
