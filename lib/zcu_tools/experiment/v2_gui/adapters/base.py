from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Mapping, Optional, Sequence, TypeVar

from zcu_tools.experiment.v2.runner import task_manager
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.progress_bar import progress_backend_scope, qt_progress_callbacks_scope

T_Config = TypeVar("T_Config", bound=Mapping[str, Any])
SchemaFieldType = Literal["int", "float", "str", "bool"]


@dataclass(frozen=True)
class ConfigFieldSchema:
    key: str
    label: str
    field_type: SchemaFieldType
    required: bool = True
    default: Any = None
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    options: Sequence[Any] = field(default_factory=tuple)


class ExperimentAdapterBase(Generic[T_Config], ABC):
    """Base adapter contract used by v2_gui experiment groups."""

    def __init__(
        self,
        exp: Any,
        *,
        scope_name: str = "v2_gui_run",
    ) -> None:
        self.exp = exp
        self.scope_name = scope_name

    def run(
        self,
        soc: Any,
        soccfg: Any,
        cfg: dict[str, Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        del soc, soccfg, cfg
        total_ref = {"total": 0}
        progress_seen = {"value": False}

        def _on_task_pbar_start(total: Optional[int], desc: str) -> None:
            del desc
            total_ref["total"] = int(total or 0)

        def _on_task_pbar_update(n: int) -> None:
            if on_progress is None:
                return
            progress_seen["value"] = True
            total = total_ref["total"] if total_ref["total"] > 0 else max(n, 1)
            on_progress(int(n), int(total))

        with task_manager.scope(self.scope_name):
            if should_cancel is not None and should_cancel():
                task_manager.cancel_current()
            with progress_backend_scope("qt"):
                with qt_progress_callbacks_scope(
                    on_start=_on_task_pbar_start,
                    on_update_to=_on_task_pbar_update,
                ):
                    self._run_exp()

        if on_progress is not None and not progress_seen["value"]:
            on_progress(1, 1)

    @abstractmethod
    def _run_exp(self) -> None:
        """Execute the wrapped experiment run."""

    @abstractmethod
    def get_config_schema(self) -> list[ConfigFieldSchema]:
        """Return normalized schema used by schema-driven panel."""

    def build_default_config(self) -> dict[str, Any]:
        """Build config defaults from schema."""
        cfg: dict[str, Any] = {}
        for item in self.get_config_schema():
            if item.default is not None:
                cfg[item.key] = item.default
        return cfg

    @abstractmethod
    def analyze(self) -> dict[str, Any]:
        """Return analysis payload for GUI text display (without figure object)."""

    @abstractmethod
    def save_run(self, filepath: Path, cfg: dict[str, Any]) -> Path:
        """Persist run payload to file."""

    @abstractmethod
    def save_analysis_figure(self, filepath: Path) -> Path:
        """Persist analysis figure and return saved path."""

    @abstractmethod
    def apply_analysis_to_context(
        self,
        analysis: dict[str, Any],
        meta_dict: MetaDict,
        module_library: ModuleLibrary,
    ) -> tuple[Path, Path]:
        """Apply analysis output to context data and return saved paths."""
