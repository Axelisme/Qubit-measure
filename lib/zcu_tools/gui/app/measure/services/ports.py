"""Narrow ports for boundaries that need isolation (ADR-0067).

Application services can collaborate through directional commands. At shared/app
or owner boundaries, a consumer may use a port instead of taking a concrete
service or infrastructure dependency. These structural ``Protocol``s declare
only what the consumer uses; an existing implementer need not inherit the port.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class CfgEdit:
    path: str
    value: object


@dataclass(frozen=True, slots=True)
class CfgEditResult:
    valid: bool
    removed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    applied: int | None = None
    actual: dict[str, object] | None = None
    errors: tuple[dict[str, str], ...] | None = None

    def to_wire(self) -> dict[str, object]:
        result: dict[str, object] = {
            "valid": self.valid,
            "removed": list(self.removed),
            "added": list(self.added),
        }
        if self.applied is not None:
            result["applied"] = self.applied
        if self.actual is not None:
            result["actual"] = self.actual
        if self.errors is not None:
            result["errors"] = list(self.errors)
        return result


if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import (
        AdapterCapabilities,
        WritebackItem,
    )
    from zcu_tools.gui.app.measure.artifact_tracker import ArtifactSectionSnapshot
    from zcu_tools.gui.app.measure.state import (
        RetiredPaneResources,
        Session,
        TabInteractionState,
    )
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.cfg.binding import CfgDraft, SettableTarget
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.plotting.figures import FigureCollection
    from zcu_tools.plotting.plots import Plots
    from zcu_tools.resources.entry.ledger_models import JsonObject

    from .persistence_types import AppPersistedState
    from .writeback import WritebackDraft


@dataclass(frozen=True, slots=True)
class RunPaneSnapshot:
    """Current Run and confirmed persistence identity.

    result is the adapter-owned record or None; source_path is the loaded file
    or None. source_operation_id identifies the publishing acquisition or None
    after Load. run_id is None until at least one persistent output succeeds;
    it survives failed resaves, but does not survive Run replacement.
    """

    result: object | None
    source_path: str | None
    source_operation_id: int | None = None
    run_id: str | None = None


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

    Render (``TabService.get_snapshot``) populates every field; persist/restore
    uses only the serializable head (adapter_name + cfg_input). cfg_input is
    detached authored Form JSON, never an accepted cfg or source snapshot. Pane-owned
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
    """One rejected tab during session restore (adapter missing / cfg invalid)."""

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
    """Outcome of applying a persisted session: how many tabs restored, and the
    per-tab rejections to surface to the user."""

    restored_tabs: int
    rejected_tabs: tuple[RestoreIssue, ...]
    # Successfully restored tabs may discard incompatible leaves, grouped by adapter.
    dropped_inputs: tuple[RestoreDroppedInputs, ...] = ()


@runtime_checkable
class PersistOriginatorPort(Protocol):
    """The Memento Originator surface the ``PersistenceCaretaker`` depends on.

    The Caretaker (a Driven Adapter doing only disk I/O) never touches State,
    services, or cfg — it only asks the originator (the Controller) for one
    immutable snapshot to write, and hands one back to restore. Two narrow
    methods keep the Caretaker decoupled from the whole Controller interface.
    """

    def capture_persisted_state(self) -> AppPersistedState: ...
    def restore_persisted_state(self, state: AppPersistedState) -> RestoreReport: ...


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
        from_dict: TabSnapshot | None = None,
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
    ) -> WritebackDraft:
        """Fix proposals, destination and native Run identity in one draft.

        run_id is historical RunMetadata.run_id, even before Save, or None
        without native evidence. Save does not change the draft's identity.
        Editor failure propagates after retiring any newly opened sessions.
        """
        ...

    def preview_draft(self, draft: Any) -> list[WritebackItem]: ...
    def teardown_draft(self, draft: Any) -> None: ...


@runtime_checkable
class CfgEditorPort(Protocol):
    """The cfg-editor surface consumed by ``WritebackService``.

    ``WritebackService`` opens a gc=False editor session for each
    module/waveform writeback item (seeded from its ``edit_schema``), tears it
    down on reanalyze/rerun, and snapshots the live draft at apply time.
    Depending on this port instead of the concrete ``CfgEditorService`` keeps
    the coupling at the interface level (ADR-0067).
    """

    def open_seeded(
        self,
        seed: CfgSchema,
        *,
        parameters: ParameterContext | None,
        gc: bool = False,
        owner_key: str | None = None,
    ) -> tuple[str, tuple[SettableTarget, ...]]: ...

    def teardown(self, editor_id: str, *, reason: str = ...) -> None: ...

    def get_draft(self, editor_id: str) -> CfgDraft: ...

    # On the port so WritebackService can apply an agent's module/waveform draft
    # edit through the item's editor session (ADR-0008): the writeback editing
    # surface internalizes editor_id, so the service writes via the port rather
    # than re-exposing the handle. Signature mirrors CfgEditorService.set_field.
    def set_field(self, editor_id: str, path: str, value: object) -> CfgEditResult: ...

    def set_fields(
        self, editor_id: str, edits: Sequence[CfgEdit], *, agent_edit: bool = False
    ) -> CfgEditResult: ...


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
