"""CfgEditorService — stateful cfg draft editing sessions shared by clients.

A session owns one ``CfgDraft`` keyed by a server-issued
``editor_id``. The service owns *every* model; a ``CfgFormWidget`` is a pluggable
viewer that ``attach``es to a model (renders + reflects it) and ``detach``es
without tearing it down — so an agent edit and a user view converge on the same
service-owned model (WYSIWYG), and a model can outlive any widget (the agent can
edit before/without a widget being open). See ADR-0008 (which records how the
headless-only and delegated-model designs were superseded) and the CfgEditor
session glossary in ``gui/CONTEXT.md``.

Lifetime is governed by ``gc`` (not two session kinds):

- **gc=True** (``open`` default — the agent builds/edits a ModuleLibrary entry):
  reclaimable on RPC disconnect and bounded by an LRU cap (orphan protection).
  ``commit`` lowers + registers into ModuleLibrary, ``discard`` drops it.
- **gc=False** (UI-owned: tab cfg / inspect / writeback): only the owner tears it
  down via ``teardown(editor_id)`` (the owning widget ``detach``es first). Not
  LRU-bounded, not reclaimed on agent disconnect. ``open_seeded`` builds one from
  an existing ``CfgSchema`` (tab cfg / writeback draft — no ml ``item_kind``, so
  it is teardown-only and rejects ``commit``).

The incremental shape is *required*, not a convenience: ModuleRef/WaveformRef
key switches rebuild the field sub-tree, so a client cannot send one complete
raw payload up-front — it must switch the ref, observe the freshly-bound paths,
then fill them. ``set_field`` returns the sub-tree rooted at the changed path
for exactly this reason.

``EvalValue`` fields (md-reference expressions, e.g. ``r_f - 0.1``) are carried
on the wire as the cfg-form tagged form ``{"__kind": "eval", "expr": ...}`` and
resolved against the live MetaDict at ``commit`` time (the app-local
``schema_to_raw_dict`` seam lowers
``EvalValue`` to its concrete ``resolved`` number), because ModuleLibrary stores
concrete numbers, never md references. ``value_ref`` tags are different: they are
resolved once at ``set_field`` time and stored only as ``DirectValue`` snapshots.

All methods run on the Qt main thread (the cfg draft and ModuleLibrary live
there); the remote service marshals handler calls accordingly.
"""

from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.cfg_binding import (
    MeasureCfgBindingHost,
    MeasureCfgBindings,
)
from zcu_tools.gui.app.main.cfg_schemas import (
    _MODULE_SPEC_FACTORIES,
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.main.specs import make_waveform_spec_by_style
from zcu_tools.gui.cfg import (
    CfgSchema,
    EvalValue,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import CfgDraft
from zcu_tools.gui.session.ports import ContextReadPort
from zcu_tools.gui.session.value_lookup import ValueLookupError, decode_value_ref

from .ports import ContextWritePort
from .remote.path_resolver import (
    ValueRefResolver,
    list_settable_paths_full,
    resolve_and_set,
)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

logger = logging.getLogger(__name__)

_ITEM_KINDS = ("module", "waveform")

# Listener for per-session push notifications: (editor_id, event_name, payload).
# event_name ∈ {"editor_changed", "editor_closed"}. Injected by the remote
# layer; the service stays ignorant of transport.
ChangeListener = Callable[[str, str, dict], None]

# Max concurrent *headless* sessions before the oldest is evicted (orphan
# protection for agents that open without commit/discard while connected).
# Delegated (widget-owned) sessions are not counted against this cap.
_MAX_HEADLESS_EDITORS = 16


class CfgEditorHost(MeasureCfgBindingHost, ContextReadPort, ContextWritePort, Protocol):
    """Composition-root surface for ``CfgEditorService`` — the facets it needs,
    all provided by the Controller: the reactive env (``_EditorCtrl``), the
    context read port (``ContextReadPort``, to seed from_name) + context write
    port (``ContextWritePort``, used at commit), and the editor-version bump. The
    service itself decomposes this into the narrow dependencies; this
    composed protocol exists only so ``build_app_services`` can type the single
    object (the Controller) that happens to satisfy all three.
    """

    def bump_editor_version(self, editor_id: str) -> None: ...

    def drop_editor_version(self, editor_id: str) -> None: ...


class CfgEditorError(RuntimeError):
    """A CfgEditor session operation failed (unknown id, bad kind, …)."""


@dataclass
class CfgEditorSession:
    """Aggregate root: one cfg-editing draft and **all logic to operate it**.

    Holds the ``CfgDraft`` and owns the behaviour over it
    (set_field / get / validity / commit). Outside code reaches the draft only
    through this session (it is the gateway), and identifies it by ``editor_id``
    (DDD: reference aggregates by id). This replaces the old anemic dataclass
    whose behaviour lived entirely on the service (docs/adr/0008 §Aggregate Root).

    Lifecycle ownership is the *Repository's* concern (``CfgEditorService``), not
    the aggregate's; the session only carries the discriminators the Repository
    needs. Every session's ``root`` is service-owned (the widget attaches to it,
    never owns it — ADR-0008); ``gc`` governs whether the Repository may reclaim
    it automatically (LRU / disconnect) or only on the owner's explicit teardown.
    """

    editor_id: str
    draft: CfgDraft
    resolve_value_ref: ValueRefResolver
    # True: agent-opened, reclaimed by LRU / RPC-disconnect (orphan protection).
    # False: UI-owned (tab / inspect / writeback) — only the owner tears it down.
    gc: bool
    # the ml entry kind being built ("module"/"waveform"); commit uses it. None
    # for a seeded session (tab cfg / writeback draft) that is teardown-only.
    item_kind: str | None = None
    # the owner's key (a tab uses its tab_id; an inspect/writeback widget uses its
    # own key), so the owner can look its editor_id back up.
    owner_key: str | None = None
    # Monotonic open order; used for gc-LRU eviction.
    seq: int = field(default=0)
    # on_change callback bound to root for the per-session change stream; held
    # so it can be disconnected when the session ends.
    change_cb: Callable[..., None] | None = None

    # -- behaviour (the aggregate operates its own draft) ------------------

    def current_paths(self) -> list[dict[str, object]]:
        """Full flat settable-path list of the draft.

        Internal use only: the change-push payload (``_attach_change_stream``)
        and ``editor.new``'s session-create return. Agent-facing reads of the
        draft go through the nested tree (``editor.get`` / ``tab.get_cfg``),
        not this flat list.
        """
        return list_settable_paths_full(self.draft)

    def set_field(self, path: str, value: object) -> dict[str, object]:
        """Mutate one field; return draft validity + which paths a ref switch
        added/removed.

        Does NOT echo cfg content back — reading it would force a lowering pass
        that eagerly evaluates EvalValue (e.g. ``r_f - 0.1`` → a concrete
        number), an unwanted side effect for a plain edit. To see the cfg, read
        the nested tree via ``editor.get`` / ``tab.get_cfg``. A ModuleRef key
        switch rebuilds its sub-tree, so ``removed`` / ``added`` (diffed over the
        whole draft) tell the agent exactly which paths disappeared / appeared,
        so it need not re-list the whole tab after a variant switch.
        """
        before = {str(e["path"]) for e in list_settable_paths_full(self.draft)}
        try:
            resolve_and_set(
                self.draft,
                path,
                _decode_value(value),
                resolve_value_ref=self.resolve_value_ref,
            )
        except ValueLookupError as exc:
            raise CfgEditorError(str(exc)) from exc
        after = {str(e["path"]) for e in list_settable_paths_full(self.draft)}
        return {
            "valid": bool(self.draft.is_valid()),
            "removed": sorted(before - after),
            "added": sorted(after - before),
        }

    def commit_schema(self) -> CfgSchema:
        """Snapshot the draft as an **un-lowered** CfgSchema for the writer.

        ADR-0006: the session's job ends at the CfgSchema snapshot; lowering
        (EvalValue → concrete, against the live md) + register belong to
        ContextService (the single write authority). ``commit`` applies only to
        ml-entry sessions (those carrying an ``item_kind``); a seeded session
        (tab cfg / writeback draft) is teardown-only and rejects commit.
        """
        if self.item_kind is None:
            raise CfgEditorError(
                f"{self.editor_id!r} is a seeded session (no item_kind); commit "
                "applies only to ml-entry sessions"
            )
        return self.draft.snapshot()


class CfgEditorService:
    """Repository for ``CfgEditorSession`` aggregates, keyed by a server id.

    Owns the session *lifecycle* (create / look up / reap) and the cross-session
    concerns (gc-LRU, per-session change stream, version bump, and the
    md/ml-change refresh of every owned model). Per-session editing behaviour
    lives on the aggregate (``CfgEditorSession``); the Repository delegates to it.

    Dependencies (docs/adr/0008): ``env_ctrl`` is the LiveModel reactive env
    (narrow port); ``read_port`` (ContextReadPort) reads the current ml to seed
    ``from_name`` sessions; ``write_port`` (ContextWritePort) is the single ml/md
    write authority used at commit (ADR-0006 — the session no longer lowers /
    registers itself); ``version_bump`` / ``version_drop`` bump / forget the ``editor:<id>`` resource
    version (a registry-level concern since the id is Repository-assigned): bump on
    every edit (so commit's guard sees concurrent edits), drop on teardown (so a
    stale dependency on a gone session reads version 0).
    """

    def __init__(
        self,
        env_ctrl: MeasureCfgBindingHost,
        read_port: ContextReadPort,
        write_port: ContextWritePort,
        version_bump: Callable[[str], None],
        version_drop: Callable[[str], None],
        bus: EventBus,
    ) -> None:
        self._bindings = MeasureCfgBindings(env_ctrl)
        self._read = read_port
        self._write = write_port
        self._version_bump = version_bump
        self._version_drop = version_drop
        self._editors: dict[str, CfgEditorSession] = {}
        self._seq = itertools.count()
        self._listener: ChangeListener | None = None
        # ADR-0008/0004 Reaction: the service owns every cfg draft, so refresh
        # payloads map explicitly to the three narrow binding refresh operations.
        from zcu_tools.gui.session.events import (
            ContextSwitchedPayload,
            DeviceChangedPayload,
            MdChangedPayload,
            MlChangedPayload,
        )

        bus.subscribe(MdChangedPayload, lambda _payload: self.refresh_expressions())
        bus.subscribe(MlChangedPayload, lambda _payload: self.refresh_references())
        bus.subscribe(
            DeviceChangedPayload, lambda _payload: self.refresh_options("devices")
        )
        bus.subscribe(ContextSwitchedPayload, lambda _payload: self.refresh_all())

    def set_change_listener(self, listener: ChangeListener | None) -> None:
        """Inject the per-session push listener (remote layer wires this).

        The listener receives ``(editor_id, event_name, payload)`` on the Qt
        main thread (where ``on_change`` fires); it must not block.
        """
        self._listener = listener

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        item_kind: str,
        *,
        discriminator: str | None = None,
        from_name: str | None = None,
        gc: bool = True,
        owner_key: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Open an ml-entry editing session (module / waveform).

        ``gc=True`` (the agent default) makes the session reclaimable by LRU and
        on RPC-disconnect. ``gc=False`` ties teardown to an explicit owner call
        (UI dialogs that build a real ml entry). ``item_kind`` is required and is
        the commit discriminator.
        """
        if item_kind not in _ITEM_KINDS:
            raise CfgEditorError(
                f"item_kind must be one of {_ITEM_KINDS}, got {item_kind!r}"
            )
        if owner_key is not None:
            self._teardown_owner(owner_key)
        spec, value = self._initial_schema(item_kind, discriminator, from_name)
        return self._make_session(
            spec, value, item_kind=item_kind, gc=gc, owner_key=owner_key
        )

    def open_seeded(
        self,
        seed: CfgSchema,
        *,
        gc: bool = False,
        owner_key: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Open a session seeded from an existing ``CfgSchema`` (no item_kind).

        Used by UI surfaces that own a cfg draft which is *not* an ml entry — a
        tab's cfg (seed = ``State.cfg_schema``) and a writeback module/waveform
        item (seed = its ``edit_schema``). Such a session is teardown-only
        (``commit`` is rejected — there is no ml-entry to register). Defaults to
        ``gc=False`` since these are always owner-driven.
        """
        if owner_key is not None:
            self._teardown_owner(owner_key)
        return self._make_session(
            seed.spec, seed.value, item_kind=None, gc=gc, owner_key=owner_key
        )

    def _make_session(
        self,
        spec,
        value,
        *,
        item_kind: str | None,
        gc: bool,
        owner_key: str | None,
    ) -> tuple[str, list[dict[str, object]]]:
        draft = self._bindings.new_draft(CfgSchema(spec=spec, value=value))
        editor_id = self._new_id(owner_key)
        session = CfgEditorSession(
            editor_id=editor_id,
            draft=draft,
            resolve_value_ref=self._bindings.resolve_value_ref,
            gc=gc,
            item_kind=item_kind,
            owner_key=owner_key,
            seq=next(self._seq),
        )
        self._editors[editor_id] = session
        self._attach_change_stream(session)
        if gc:
            self._evict_excess_gc()
        return editor_id, session.current_paths()

    def editor_id_for_owner(self, owner_key: str) -> str | None:
        for editor_id, session in self._editors.items():
            if session.owner_key == owner_key:
                return editor_id
        return None

    def owner_of_editor(self, editor_id: str) -> str | None:
        """The owner_key a session is keyed to (tab_id for a tab cfg draft).

        ``None`` for unknown sessions or owner-less (gc) sessions. Lets a caller
        that knows run-state (the Controller façade) gate edits on the owning
        tab without this service needing to know about runs.
        """
        session = self._editors.get(editor_id)
        return session.owner_key if session is not None else None

    def get_draft(self, editor_id: str) -> CfgDraft:
        """Return the service-owned ``CfgDraft`` for ``editor_id``.

        Exposed so the owning widget can ``attach`` to a service-owned model
        (ADR-0008). The widget renders + reflects this model but never owns its
        lifetime (the service does).
        """
        return self._require(editor_id).draft

    def set_field(self, editor_id: str, path: str, value: object) -> dict[str, object]:
        return self._require(editor_id).set_field(path, value)

    def commit(self, editor_id: str, name: str) -> None:
        # ADR-0006: the aggregate yields its un-lowered CfgSchema; ContextService
        # (the single write authority) lowers + registers. The Repository owns
        # teardown only — after a successful write, so a validation failure
        # (raised by the write port) leaves the draft intact for the agent to fix.
        session = self._require(editor_id)
        schema = session.commit_schema()
        if session.item_kind == "module":
            self._write.set_ml_module_from_schema(name, schema)
        else:
            self._write.set_ml_waveform_from_schema(name, schema)
        self._remove(editor_id, teardown=True, reason="committed")

    def discard(self, editor_id: str) -> None:
        """Drop a session, tearing down its LiveModel.

        The agent-facing discard of an ml-entry draft (discard = "throw the draft
        away"). Unknown id raises (the agent referenced a gone session), unlike
        the lenient ``teardown`` which is a no-op for owner-teardown races.
        """
        self._require(editor_id)
        self._remove(editor_id, teardown=True, reason="discarded")

    def teardown(self, editor_id: str, *, reason: str = "tab_closed") -> None:
        """Tear down a UI-owned (``gc=False``) session and its LiveModel.

        Called by the owner (tab close, dialog close, writeback reanalyze) — the
        service owns every model now (ADR-0008), so it is also the one to tear it
        down; the widget only ``detach``es first. Unknown id is a no-op (teardown
        may race a never-opened owner).
        """
        if editor_id not in self._editors:
            return
        self._remove(editor_id, teardown=True, reason=reason)

    def _teardown_owner(self, owner_key: str) -> None:
        """Tear down the existing session for ``owner_key`` if any (re-open)."""
        existing = self.editor_id_for_owner(owner_key)
        if existing is not None:
            self._remove(existing, teardown=True, reason="reopened")

    def discard_for_client(self, editor_ids: list[str]) -> None:
        """Reclaim a batch of *gc* sessions (per-connection cleanup).

        UI-owned (``gc=False``) sessions are owner-driven and never reclaimed on
        RPC disconnect, so they are skipped even if their id appears here.
        Unknown ids are ignored.
        """
        for editor_id in editor_ids:
            session = self._editors.get(editor_id)
            if session is not None and session.gc:
                self._remove(editor_id, teardown=True, reason="disconnected")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_id(self, owner_key: str | None) -> str:
        # Owner-keyed sessions (a tab cfg draft / writeback item) read as
        # '<owner>-ed' so the agent sees which tab/item an editor_id belongs to
        # without a lookup; the owner_key is already unique. Owner-less sessions
        # (agent-opened ml-entry edits) keep an opaque short id.
        if owner_key:
            return f"{owner_key}-ed"
        return "editor-" + uuid.uuid4().hex[:8]

    def _attach_change_stream(self, session: CfgEditorSession) -> None:
        """Bind a root ``on_change`` callback that pushes ``editor_changed``.

        Fires on any descendant edit (root on_change bubbles). The payload
        carries the full current path list (root on_change does not say which
        path changed); subtree-only optimisation can come later.

        Symmetric teardown: the callback stored in ``session.change_cb`` is
        disconnected in ``_remove`` (the single teardown sink) before the session
        is dropped — never leave a dangling hook on a torn-down root.
        """
        editor_id = session.editor_id

        def _on_change(*_: object) -> None:
            # The draft for this session was just written (main thread); bump its
            # version so editor.commit's guard can detect a concurrent edit.
            self._version_bump(editor_id)
            self._emit(
                editor_id,
                "editor_changed",
                {"paths": session.current_paths()},
            )

        session.change_cb = _on_change
        session.draft.on_change.connect(_on_change)

    def _emit(self, editor_id: str, event_name: str, payload: dict) -> None:
        if self._listener is None:
            return
        try:
            self._listener(editor_id, event_name, payload)
        except Exception:  # pragma: no cover — listener must not break the edit
            logger.exception("cfg-editor change listener raised for %s", editor_id)

    def _remove(self, editor_id: str, *, teardown: bool, reason: str) -> None:
        session = self._editors.pop(editor_id, None)
        if session is None:
            return
        if session.change_cb is not None:
            session.draft.on_change.disconnect(session.change_cb)
            session.change_cb = None
        if teardown:
            session.draft.close()
        # Forget the session's resource version (symmetric to tab/device drop):
        # a later stale dependency on this gone editor reads version 0 and the
        # guard treats it as stale, rather than spuriously matching a retained
        # version. Done whether or not we tear the root down — the session is gone.
        self._version_drop(editor_id)
        # Notify subscribers the session is gone (after state is consistent).
        self._emit(editor_id, "editor_closed", {"reason": reason})

    def _evict_excess_gc(self) -> None:
        gc_sessions = [(eid, s) for eid, s in self._editors.items() if s.gc]
        if len(gc_sessions) <= _MAX_HEADLESS_EDITORS:
            return
        # Evict oldest-first until back under the cap.
        gc_sessions.sort(key=lambda item: item[1].seq)
        for eid, _ in gc_sessions[: len(gc_sessions) - _MAX_HEADLESS_EDITORS]:
            self._remove(eid, teardown=True, reason="evicted")

    # ------------------------------------------------------------------
    # External refresh (md/ml change → refresh every owned model's EvalValue)
    # ------------------------------------------------------------------

    def refresh_expressions(self) -> None:
        for session in list(self._editors.values()):
            session.draft.refresh_expressions()

    def refresh_options(self, source_id: str | None = None) -> None:
        for session in list(self._editors.values()):
            session.draft.refresh_options(source_id)

    def refresh_references(self, kind: str | None = None) -> None:
        for session in list(self._editors.values()):
            session.draft.refresh_references(kind)

    def refresh_all(self) -> None:
        for session in list(self._editors.values()):
            session.draft.refresh_expressions()
            session.draft.refresh_options()
            session.draft.refresh_references()

    def _require(self, editor_id: str) -> CfgEditorSession:
        session = self._editors.get(editor_id)
        if session is None:
            raise CfgEditorError(f"unknown editor session: {editor_id!r}")
        return session

    def _initial_schema(
        self,
        item_kind: str,
        discriminator: str | None,
        from_name: str | None,
    ):
        if from_name is not None:
            ml = self._read.get_current_ml()
            if item_kind == "module":
                if from_name not in ml.modules:
                    raise CfgEditorError(f"unknown module: {from_name!r}")
                return module_cfg_to_value(ml.modules[from_name])
            if from_name not in ml.waveforms:
                raise CfgEditorError(f"unknown waveform: {from_name!r}")
            return waveform_cfg_to_value(ml.waveforms[from_name])

        if discriminator is None:
            raise CfgEditorError("either 'discriminator' or 'from_name' is required")
        if item_kind == "module":
            factory = _MODULE_SPEC_FACTORIES.get(discriminator)
            if factory is None:
                raise CfgEditorError(
                    f"unknown module type {discriminator!r}; "
                    f"allowed: {sorted(_MODULE_SPEC_FACTORIES)}"
                )
            spec = factory()
        else:
            try:
                spec = make_waveform_spec_by_style(discriminator)
            except (KeyError, RuntimeError) as exc:
                raise CfgEditorError(
                    f"unknown waveform style {discriminator!r}: {exc}"
                ) from exc
        return spec, make_default_value(spec)


def _decode_value(value: object) -> object:
    """Turn tagged values into model-layer value objects; pass others through.

    The agent sends ``{"__kind": "eval", "expr": "..."}`` for an md-reference
    expression (the same tag used by the cfg-form codec), or
    ``{"__kind": "value_ref", "key": "...", "type": "float"}`` for a registered
    value source. Everything else is a plain JSON scalar that ``resolve_and_set``
    handles directly.
    """
    if isinstance(value, dict) and value.get("__kind") == "eval":
        expr = value.get("expr")
        if not isinstance(expr, str):
            raise CfgEditorError("eval value requires a string 'expr'")
        return EvalValue(expr=expr)
    try:
        ref = decode_value_ref(value)
    except (ValueError, ValueLookupError) as exc:
        raise CfgEditorError(str(exc)) from exc
    if ref is not None:
        return ref
    return value
