"""Narrow ports for boundaries that need isolation (ADR-0067).

Application services can collaborate through directional commands. At shared/app
or owner boundaries, a consumer may use a port instead of taking a concrete
service or infrastructure dependency. These structural ``Protocol``s declare
only what the consumer uses; an existing implementer need not inherit the port.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)


@dataclass(frozen=True, slots=True)
class SaveDataSubmission:
    """Accepted save work; the reserved path is not proof of successful I/O."""

    operation_id: int
    data_path: str


@dataclass(frozen=True, slots=True)
class SaveDestination:
    kind: ArtifactKind
    path: str


@dataclass(frozen=True, slots=True)
class SaveArtifactsSubmission:
    """Reserved destinations; only a finished outcome establishes successful I/O."""

    operation_id: int
    destinations: tuple[SaveDestination, ...]


@dataclass(frozen=True, slots=True)
class ActiveSaveOperation:
    operation_id: int
    tab_id: str


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
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.measure.adapter import (
        AdapterCapabilities,
        WritebackItem,
    )
    from zcu_tools.gui.app.measure.artifact_tracker import (
        ArtifactKind,
        ArtifactSnapshot,
    )
    from zcu_tools.gui.app.measure.state import (
        RetiredPaneResources,
        Session,
        TabInteractionState,
    )
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.cfg.binding import CfgDraft, SettableTarget
    from zcu_tools.gui.session.types import SessionEnv

    from .persistence_types import AppPersistedState


@dataclass(frozen=True, slots=True)
class PathResourceSnapshot:
    """Read model for one independently-owned path resource."""

    override: str | None
    path: str | None


@dataclass(frozen=True, slots=True)
class RunPaneSnapshot:
    result: object | None
    source_path: str | None


@dataclass(frozen=True, slots=True)
class AnalysisPaneSnapshot:
    params: object | None
    result: object | None
    figure: Figure | None
    writeback_items: tuple[WritebackItem, ...]
    image_path: PathResourceSnapshot
    has_writeback_draft: bool = False
    source_operation_id: int | None = None


@dataclass(frozen=True, slots=True)
class PostAnalysisPaneSnapshot:
    params: object | None
    result: object | None
    figure: Figure | None
    writeback_items: tuple[WritebackItem, ...]
    image_path: PathResourceSnapshot
    has_writeback_draft: bool = False
    source_operation_id: int | None = None


@dataclass(frozen=True, slots=True)
class SavePaneSnapshot:
    data_path: PathResourceSnapshot
    comment: str = ""


@dataclass(frozen=True, slots=True)
class TabPathsSnapshot:
    """The three path resources projected independently by a tab snapshot."""

    data: PathResourceSnapshot
    analysis_image: PathResourceSnapshot
    post_analysis_image: PathResourceSnapshot


@dataclass(frozen=True)
class TabSnapshot:
    """Immutable full-state snapshot of one tab (contract-layer DTO).

    Render (``TabService.get_snapshot``) populates every field; persist/restore
    uses only the serializable head (adapter_name + cfg_schema). Pane-owned
    read models (run/analysis/post_analysis/save/paths) are the only
    runtime/read-model contract — no flat projections for legacy callers.
    """

    adapter_name: str
    cfg_schema: CfgSchema
    tab_id: str | None = None
    interaction: TabInteractionState | None = None
    capabilities: AdapterCapabilities | None = None
    run: RunPaneSnapshot | None = None
    analysis: AnalysisPaneSnapshot | None = None
    post_analysis: PostAnalysisPaneSnapshot | None = None
    save: SavePaneSnapshot | None = None
    paths: TabPathsSnapshot | None = None
    artifacts: tuple[ArtifactSnapshot, ...] = ()


@dataclass(frozen=True)
class RestoreIssue:
    """One rejected tab during session restore (adapter missing / cfg invalid)."""

    subject: str
    message: str


@dataclass(frozen=True)
class RestoreReport:
    """Outcome of applying a persisted session: how many tabs restored, and the
    per-tab rejections to surface to the user."""

    restored_tabs: int
    rejected_tabs: tuple[RestoreIssue, ...]


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
    """The single authority for ml/md content writes (ADR-0067).

    Sources holding an un-lowered ``CfgSchema`` (editor commit, writeback apply,
    inspect save, create_from_role) write through this port; ContextService
    lowers (app-local ``schema_to_raw_dict`` with the live md, so callers cannot
    forget md)
    + registers, and on success bumps the ``context`` version + emits
    ML/MD_CHANGED. The only implementer is ContextService.

    ``apply_writes`` is the batch entry: a successful apply (writeback) of md
    attrs + multiple ml entries lands as **one** version bump and **at most one**
    ML_CHANGED + one MD_CHANGED (the per-write methods each bump/emit on their
    own; batching avoids N redundant full-refreshes). A failed batch is not
    rolled back and publishes nothing; see ``ContextWrites``.
    """

    def set_ml_module_from_schema(self, name: str, schema: CfgSchema) -> None: ...
    def set_ml_waveform_from_schema(self, name: str, schema: CfgSchema) -> None: ...
    def replace_ml_module_from_schema(
        self, old_name: str, new_name: str, schema: CfgSchema
    ) -> None: ...
    def replace_ml_waveform_from_schema(
        self, old_name: str, new_name: str, schema: CfgSchema
    ) -> None: ...
    def set_md_attr(self, key: str, value: Any) -> None: ...
    def apply_writes(self, writes: ContextWrites) -> None: ...


@dataclass(frozen=True)
class ContextWrites:
    """A batch of ml/md content writes applied in insertion order.

    On success the owner bumps the context version once and emits one event per
    touched kind. The batch is not all-or-nothing: a later lower, register or
    dump failure leaves the earlier live md/ml changes in place without a bump
    or event. ``md`` maps attr name → value; ``ml_modules`` / ``ml_waveforms``
    map entry name → its un-lowered ``CfgSchema``."""

    md: dict[str, Any]
    ml_modules: dict[str, CfgSchema]
    ml_waveforms: dict[str, CfgSchema]


@runtime_checkable
class TabLifecyclePort(Protocol):
    """Tab create/restore/close + cfg as commanded by ``WorkspaceService``.

    ``WorkspaceService`` orchestrates the tab lifecycle (one-way command); it
    depends on this narrow port to isolate the tab lifecycle command surface
    across the workspace/tab owner boundary (ADR-0067). This is a directional
    service command, not a ban on service collaboration.
    """

    def new_tab(
        self, adapter_name: str, from_dict: TabSnapshot | None = None
    ) -> str: ...
    def close_tab(self, tab_id: str) -> None: ...
    def make_default_cfg(self, adapter_name: str) -> CfgSchema: ...


@runtime_checkable
class WritebackLifecyclePort(Protocol):
    """Writeback lifecycle surface used by result-owning panes.

    The pane-owned opaque draft is the only contract; tab-level adapters are
    removed.
    """

    def create_draft(self, items: Iterable[WritebackItem]) -> Any: ...
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
class TabResultWritePort(Protocol):
    """The narrow State-write contract a run policy depends on (ADR-0066).

    Run's lifecycle writes only these three tab-result mutations; depending on
    this port instead of the concrete ``State`` keeps the policy bound to a
    contract, not behaviour (ADR-0067). ``State`` is the only implementer and
    satisfies it structurally (no inheritance change)."""

    def clear_tab_results(self, tab_id: str) -> RetiredPaneResources: ...
    def set_tab_running(self, tab_id: str, running: bool) -> None: ...
    def update_tab_result(
        self, tab_id: str, result: object
    ) -> RetiredPaneResources: ...


@runtime_checkable
class TabAnalyzeWritePort(Protocol):
    """The narrow State-write contract an analyze / post-analyze policy depends
    on (ADR-0066). Same rationale as ``TabResultWritePort``; ``State`` is the
    only implementer. Result replacement methods return detached resources for
    post-commit draft cleanup."""

    def set_tab_analyzing(self, tab_id: str, analyzing: bool) -> None: ...
    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        figure: Figure | None,
        writeback_draft: object | None = None,
        analyze_params_instance: object = ...,
        *,
        source_operation_id: int | None = None,
    ) -> RetiredPaneResources: ...
    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        figure: Figure | None,
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
class RunStatePort(TabBusyQueryPort, TabResultWritePort, Protocol):
    """Complete State surface consumed by ``RunService``."""


@runtime_checkable
class AnalyzeStatePort(
    TabBusyQueryPort,
    TabAnalyzeReadPort,
    TabAnalyzeWritePort,
    Protocol,
):
    """Complete State surface consumed by analyze and post-analyze services."""
