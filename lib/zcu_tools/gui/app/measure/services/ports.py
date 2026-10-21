"""Narrow ports for boundaries that need isolation (ADR-0067).

Application services can collaborate through directional commands. At shared/app
or owner boundaries, a consumer may use a port instead of taking a concrete
service or infrastructure dependency. These structural ``Protocol``s declare
only what the consumer uses; an existing implementer need not inherit the port.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from pydantic import JsonValue

from zcu_tools.gui.form import FormPath
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.gui.session.types import ParameterContext

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
    from zcu_tools.gui.app.measure.artifacts import ArtifactSectionSnapshot
    from zcu_tools.gui.app.measure.state import (
        RetiredPaneResources,
        Session,
        TabInteractionState,
    )
    from zcu_tools.gui.app.measure.writeback import WritebackItem
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.plotting.figures import FigureCollection
    from zcu_tools.plotting.plots import Plots
    from zcu_tools.resources.entry.ledger_models import JsonObject

    from ..writeback import WritebackDraft
    from .tab_input import TabInput
    from .workspace_types import WorkspaceTab


@dataclass(frozen=True, slots=True)
class RunPaneSnapshot:
    """Current Run and confirmed persistence identity.

    result is the adapter-owned record or None; source_path is the loaded file
    or None. source_operation_id identifies the publishing acquisition or None
    after Load. run_id is None until at least one persistent output succeeds;
    it survives failed resaves, but does not survive Run replacement.
    acquisition_run_id is the fixed Run/Load metadata identity before any Save;
    None means no native acquisition metadata, not unsaved data.
    """

    result: object | None
    source_path: str | None
    source_operation_id: int | None = None
    run_id: str | None = None
    acquisition_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisPaneSnapshot:
    params: object | None
    result: object | None
    figures: FigureCollection | None
    writeback_items: tuple[WritebackItem, ...]
    has_writeback_draft: bool = False
    source_operation_id: int | None = None
    result_params: object | None = None
    # Fixed reasons for proposals omitted because Run-time sample evidence is missing.
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PostAnalysisPaneSnapshot:
    params: object | None
    result: object | None
    figures: FigureCollection | None
    writeback_items: tuple[WritebackItem, ...]
    has_writeback_draft: bool = False
    source_operation_id: int | None = None
    result_params: object | None = None
    # Fixed reasons for proposals omitted because Run-time sample evidence is missing.
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SavePaneSnapshot:
    """Current annotation and how the Data tab presents this tab's outputs.

    comment is the annotation draft. display_bases are the directories output
    paths are shown relative to: the parents of the Run's own result and
    database roots (so "Database/<entry>/runs/…" and "result/<entry>/…"), or
    () when the Run has no saved-point storage. saves_to_entry is False when
    the adapter declares no persistence (experiment_tag is None), so the Data
    tab can say why nothing is saveable. Destinations belong to outputs.
    """

    comment: str = ""
    display_bases: tuple[Path, ...] = ()
    saves_to_entry: bool = True


@dataclass(frozen=True)
class TabSnapshot:
    """Immutable full-state snapshot of one tab (contract-layer DTO).

    Render (``TabService.get_snapshot``) populates every field. cfg_input is
    detached raw input for live observation, not persistence wire JSON. Restore
    uses WorkspaceTab with the shared FormInput codec instead. Pane-owned
    read models (run/analysis/post_analysis/save/artifacts) are the only
    runtime/read-model contract — no flat projections for legacy callers.
    """

    adapter_name: str
    cfg_input: dict[str, JsonValue]
    tab_id: str | None = None
    interaction: TabInteractionState | None = None
    capabilities: AdapterCapabilities | None = None
    run: RunPaneSnapshot | None = None
    analysis: AnalysisPaneSnapshot | None = None
    post_analysis: PostAnalysisPaneSnapshot | None = None
    save: SavePaneSnapshot | None = None
    artifacts: tuple[ArtifactSectionSnapshot, ...] = ()


@dataclass(frozen=True)
class RestoreIssue:
    """One rejected tab during live workspace reload (adapter missing / cfg invalid)."""

    subject: str
    message: str


@dataclass(frozen=True)
class RestoreDroppedInputs:
    """Compatible restoration kept a tab but discarded these authored input paths.

    adapter_name identifies the persisted experiment. paths are model-field paths,
    with integer indexes when FormState reports them; they are never wire strings.
    """

    adapter_name: str
    paths: tuple[FormPath, ...]


@dataclass(frozen=True)
class RestoreReport:
    """Outcome of live RAM workspace reload: how many tabs restored, and the
    per-tab rejections to surface to the user."""

    restored_tabs: int
    rejected_tabs: tuple[RestoreIssue, ...]
    # Successfully restored tabs may discard incompatible leaves, grouped by adapter.
    dropped_inputs: tuple[RestoreDroppedInputs, ...] = ()


@runtime_checkable
class ContextWritePort(Protocol):
    """App library structure commands carrying persistent Specs.

    ParameterContext fixes the destination and expression/reference data.
    ContextService owns the resource transaction and publication; no live
    library handle, batch writeback or implicit target is exposed here.
    """

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        """Apply one typed structure command; target/Spec/write errors propagate."""
        ...


@runtime_checkable
class TabLifecyclePort(Protocol):
    """Tab create/restore/close + cfg as commanded by ``WorkspaceService``.

    ``WorkspaceService`` orchestrates the tab lifecycle (one-way command); it
    depends on this narrow port to isolate the tab lifecycle command surface
    across the workspace/tab owner boundary (ADR-0067). This is a directional
    service command, not a ban on service collaboration.
    """

    def new_tab(
        self,
        adapter_name: str,
        from_dict: WorkspaceTab | None = None,
        *,
        input: TabInput | None = None,
    ) -> str:
        """Create or restore a tab using the current session sources."""
        ...

    def close_tab(self, tab_id: str) -> None: ...
    def dropped_input_paths(self, tab_id: str) -> tuple[FormPath, ...]:
        """Return paths discarded when creating this tab from saved Form input."""
        ...


@runtime_checkable
class WritebackLifecyclePort(Protocol):
    """Writeback lifecycle surface used by result-owning panes.

    The pane-owned opaque draft is the only contract; tab-level adapters are
    removed.
    """

    def create_draft(
        self,
        items: Iterable[WritebackItem],
        *,
        parameters: ParameterContext | None,
        experiment_name: str,
        run_id: str | None = None,
        device_snapshot: Mapping[str, JsonObject] | None = None,
        sample_device_alias: str | None = None,
    ) -> WritebackDraft:
        """Fix proposals, destination and native Run identity in one draft.

        run_id is historical RunMetadata.run_id, even before Save, or None
        without native evidence. Save does not change the draft's identity.
        sample_device_alias is the adapter's coordinate device alias ("$name").
        None omits only Sample proposals. Resolve it from parameters and the
        fixed device_snapshot; missing facts produce public unavailable reasons.
        A module proposal that is not a concrete persistent Spec raises.
        """
        ...

    def preview_draft(self, draft: Any) -> list[WritebackItem]: ...
    def teardown_draft(self, draft: Any) -> None: ...


@runtime_checkable
class TabAnalyzeWritePort(Protocol):
    """The narrow State-write contract an analyze / post-analyze policy depends
    on (ADR-0066). Result replacement methods return detached resources for
    post-commit draft cleanup."""

    def set_tab_analyzing(self, tab_id: str, *, analyzing: bool) -> None: ...
    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        plots: Plots | None,
        writeback_draft: object | None = None,
        analyze_params_instance: object = ...,
        *,
        source_operation_id: int | None = None,
    ) -> RetiredPaneResources: ...
    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        plots: Plots | None,
        *,
        post_analyze_params_instance: object = ...,
        writeback_draft: object | None = None,
        source_operation_id: int | None = None,
    ) -> RetiredPaneResources: ...


@runtime_checkable
class TabBusyQueryPort(Protocol):
    """Dynamic per-tab busy query shared by run/analyze operation boundaries."""

    def is_tab_busy(self, tab_id: str) -> bool: ...


@runtime_checkable
class TabAnalyzeReadPort(Protocol):
    """The narrow State-read contract an analyze / post-analyze policy needs.

    Analyze services need the immutable facts required to build request objects
    at the operation boundary: the current tab and the active experiment context.
    They do not get the rest of ``State`` through their type contract.
    """

    @property
    def session_env(self) -> SessionEnv: ...

    def get_tab(self, tab_id: str) -> Session[Any, Any, Any, Any]: ...


@runtime_checkable
class AnalyzeStatePort(
    TabBusyQueryPort,
    TabAnalyzeReadPort,
    TabAnalyzeWritePort,
    Protocol,
):
    """Complete State surface consumed by analyze and post-analyze services."""
