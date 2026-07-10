"""Ports — interfaces that application services depend on instead of concrete
infrastructure (driven/secondary adapters).

DDD/Hexagonal: an application service must not depend on a concrete external
system (persistence, project file I/O, hardware driver); it depends on a *port*
(an interface) which a driven adapter implements. See ``docs/adr/0008`` §
"Driven Adapter" and the M1 milestone.

These ports are ``Protocol``s (structural), so the existing concrete services
(``StartupPersistenceService`` / ``SessionPersistenceService`` / ``IOManager``)
satisfy them without any inheritance change — M1 only narrows what each consumer
*sees* and lets tests inject in-memory fakes. Each port declares exactly the
methods its consumer calls (interface segregation), so a consumer cannot reach
infrastructure capability it has no business using.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import (
        AdapterCapabilities,
        CfgSchema,
        SavePaths,
        WritebackItem,
    )
    from zcu_tools.gui.app.main.state import Session, TabInteractionState
    from zcu_tools.gui.cfg.binding import CfgDraft
    from zcu_tools.gui.session.types import ExpContext

    from .persistence_types import AppPersistedState


@dataclass(frozen=True)
class TabSnapshot:
    """Immutable full-state snapshot of one tab (contract-layer DTO).

    A single type for two consumers (formerly ``TabViewSnapshot`` + the on-disk
    ``PersistedTab``):

    - **render** (``TabService.get_snapshot``): every field populated, handed to
      the View to draw one tab.
    - **restore** (``TabService.new_tab(from_dict=...)``): only the serializable
      head fields carry meaning; the live fields below are ``None`` / empty.

    ``cfg_schema`` is always the *live* ``CfgSchema`` (resolved EvalValue), which
    the render path uses directly. The disk codec (``SessionPersistenceService``)
    converts ``cfg_schema`` ↔ raw at the file boundary, so the persisted form
    never leaks into the snapshot. Lives in ``ports`` (the contract layer) so an
    application service can pass it around without importing a sibling
    application-service module (ADR-0005).
    """

    adapter_name: str
    cfg_schema: CfgSchema
    # The user's explicit override only (None = follow the adapter suggestion).
    # This is the serializable save-path state — persist/restore round-trip it so
    # a reload never pins an adapter-derived path.
    save_paths_override: SavePaths | None
    # Live render-only fields; None / empty on the persist + restore paths.
    tab_id: str | None = None
    interaction: TabInteractionState | None = None
    capabilities: AdapterCapabilities | None = None
    analyze_params: object | None = None
    # Post-analysis (second layer) live render fields, mirroring analyze_params /
    # figure. None until the user has a post param instance / a post result.
    post_analyze_params: object | None = None
    post_figure: Figure | None = None
    writeback_items: tuple[WritebackItem, ...] = ()
    figure: Figure | None = None
    # Render-computed effective paths (override, else adapter suggestion from
    # ctx). The View shows this; it is *not* persisted (derivable on restore).
    save_paths: SavePaths | None = None
    # Source file for a loaded result. None for live run results and restored
    # app-state snapshots; shares the tab:<id>:result lifetime.
    result_source_path: str | None = None


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
    """The single authority for ml/md content writes (ADR-0006).

    Sources holding an un-lowered ``CfgSchema`` (editor commit, writeback apply,
    inspect save, create_from_role) write through this port; ContextService
    lowers (app-local ``schema_to_raw_dict`` with the live md, so callers cannot
    forget md)
    + registers + bumps the ``context`` version + emits ML/MD_CHANGED. The only
    implementer is ContextService.

    ``apply_writes`` is the batch entry: a single apply (writeback) of md attrs +
    multiple ml entries lands as **one** version bump and **at most one**
    ML_CHANGED + one MD_CHANGED (the per-write methods each bump/emit on their
    own; batching avoids N redundant full-refreshes).
    """

    def set_ml_module_from_schema(self, name: str, schema: CfgSchema) -> None: ...
    def set_ml_waveform_from_schema(self, name: str, schema: CfgSchema) -> None: ...
    def set_md_attr(self, key: str, value: Any) -> None: ...
    def apply_writes(self, writes: ContextWrites) -> None: ...


@dataclass(frozen=True)
class ContextWrites:
    """A batch of ml/md content writes applied atomically (one bump + one emit
    per kind). ``md`` maps attr name → value; ``ml_modules`` / ``ml_waveforms``
    map entry name → its un-lowered ``CfgSchema``. Insertion order preserved."""

    md: dict[str, Any]
    ml_modules: dict[str, CfgSchema]
    ml_waveforms: dict[str, CfgSchema]


@runtime_checkable
class WritebackQueryPort(Protocol):
    """The writeback-items query as used by the tab read model.

    ``TabService.get_snapshot`` is a read-model assembler; it composes a tab's
    writeback proposals into the snapshot but must not depend on the concrete
    ``WritebackService`` (ADR-0005 violation 2 — no app-service→app-service
    coupling). It depends on this narrow query port instead, which prevents a
    back-edge from ever forming. ``WritebackService`` implements it.
    """

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]: ...


@runtime_checkable
class TabLifecyclePort(Protocol):
    """Tab create/restore/close + cfg as commanded by ``WorkspaceService``.

    ``WorkspaceService`` orchestrates the tab lifecycle (one-way command); it
    depends on this port, not the concrete ``TabService`` (ADR-0005 violation 2).
    """

    def new_tab(
        self, adapter_name: str, from_dict: TabSnapshot | None = None
    ) -> str: ...
    def close_tab(self, tab_id: str) -> None: ...
    def make_default_cfg(self, adapter_name: str) -> CfgSchema: ...


@runtime_checkable
class WritebackLifecyclePort(Protocol):
    """The writeback lifecycle surface consumed by ``RunService`` and
    ``AnalyzeService``.

    These services need to tear down per-tab writeback editor models before
    clearing / replacing a run result, and to compute a fresh draft once an
    analyze finishes (ADR-0008). Depending on this port instead of the concrete
    ``WritebackService`` keeps the coupling at the interface level (ADR-0005).
    """

    def teardown_tab_items(self, tab_id: str) -> None: ...

    def compute_items_for_tab(
        self, tab_id: str, analyze_result: Any
    ) -> list[WritebackItem]: ...


@runtime_checkable
class CfgEditorPort(Protocol):
    """The cfg-editor surface consumed by ``WritebackService``.

    ``WritebackService`` opens a gc=False editor session for each
    module/waveform writeback item (seeded from its ``edit_schema``), tears it
    down on reanalyze/rerun, and snapshots the live draft at apply time.
    Depending on this port instead of the concrete ``CfgEditorService`` keeps
    the coupling at the interface level (ADR-0005).
    """

    def open_seeded(
        self,
        seed: CfgSchema,
        *,
        gc: bool = False,
        owner_key: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]: ...

    def teardown(self, editor_id: str, *, reason: str = ...) -> None: ...

    def get_draft(self, editor_id: str) -> CfgDraft: ...

    # On the port so WritebackService can apply an agent's module/waveform draft
    # edit through the item's editor session (ADR-0008): the writeback editing
    # surface internalizes editor_id, so the service writes via the port rather
    # than re-exposing the handle. Signature mirrors CfgEditorService.set_field.
    def set_field(
        self, editor_id: str, path: str, value: object
    ) -> dict[str, object]: ...


@runtime_checkable
class TabResultWritePort(Protocol):
    """The narrow State-write contract a run policy depends on (ADR-0026 §3).

    Run's lifecycle writes only these three tab-result mutations; depending on
    this port instead of the concrete ``State`` keeps the policy bound to a
    contract, not behaviour (ADR-0005). ``State`` is the only implementer and
    satisfies it structurally (no inheritance change)."""

    def clear_tab_results(self, tab_id: str) -> None: ...
    def set_tab_running(self, tab_id: str, running: bool) -> None: ...
    def update_tab_result(self, tab_id: str, result: object) -> None: ...


@runtime_checkable
class TabAnalyzeWritePort(Protocol):
    """The narrow State-write contract an analyze / post-analyze policy depends
    on (ADR-0026 §3). Same rationale as ``TabResultWritePort``; ``State`` is the
    only implementer. ``update_tab_analyze``'s ``writeback_items`` keyword
    mirrors State exactly so the structural match holds."""

    def set_tab_analyzing(self, tab_id: str, analyzing: bool) -> None: ...
    def update_tab_analyze(
        self,
        tab_id: str,
        analyze_result: object,
        figure: Figure | None,
        writeback_items: list[WritebackItem] | None = None,
    ) -> None: ...
    def update_tab_post_analyze(
        self,
        tab_id: str,
        post_analyze_result: object,
        figure: Figure | None,
    ) -> None: ...


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
    def exp_context(self) -> ExpContext: ...

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
