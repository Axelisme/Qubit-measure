"""CfgEditorService — stateful LiveModel editing sessions shared by clients.

A session owns one ``SectionLiveField`` (a cfg draft) keyed by a server-issued
``editor_id``. It is the shared **draft SSOT** that multiple clients address:
a tab's ``CfgFormWidget`` and an MCP agent both hold the same ``editor_id`` and
operate the same session. See ``docs/adr/0003-shared-cfg-editor-session.md`` and
the CfgEditor session glossary in ``gui/CONTEXT.md``.

Two session kinds differ only by *who owns the lifetime*:

- **headless** (``open``): the agent builds/edits a ModuleLibrary entry with no
  widget attached. ``commit`` lowers + registers into ModuleLibrary, ``discard``
  drops it; both tear down the session's LiveModel. Reclaimed on RPC disconnect
  and bounded by an LRU cap (orphan protection).
- **delegated** (``register_delegated_session``): a ``CfgFormWidget`` (tab,
  inspect dialog or writeback) already owns a live LiveModel; the service
  registers *that same instance* (no new tree — the shared instance is what
  makes WYSIWYG work) and returns an ``editor_id``. The lifetime is the owner's:
  ``close`` removes the registration but does **not** tear down the root (the
  widget's ``cfg_form.clear()`` owns that). Not LRU-bounded, not reclaimed on
  agent disconnect. Only a tab's editor_id is currently exposed to agents (via
  ``tab.snapshot``); inspect/writeback sessions are registered for uniformity
  (consistent change stream) but not yet addressable from the wire.

The incremental shape is *required*, not a convenience: ModuleRef/WaveformRef
key switches rebuild the field sub-tree, so a client cannot send one complete
raw payload up-front — it must switch the ref, observe the freshly-bound paths,
then fill them. ``set_field`` returns the sub-tree rooted at the changed path
for exactly this reason.

``EvalValue`` fields (md-reference expressions, e.g. ``r_f - 0.1``) are carried
on the wire as the cfg-form tagged form ``{"__kind": "eval", "expr": ...}`` and
resolved against the live MetaDict at ``commit`` time (``schema_to_dict`` lowers
``EvalValue`` to its concrete ``resolved`` number), because ModuleLibrary stores
concrete numbers, never md references.

All methods run on the Qt main thread (the LiveModel and ModuleLibrary live
there); the remote service marshals handler calls accordingly.
"""

from __future__ import annotations

import itertools
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from zcu_tools.gui.adapter import (
    CfgSchema,
    EvalValue,
    make_default_value,
    schema_to_dict,
)
from zcu_tools.gui.cfg_schemas import (
    _MODULE_SPEC_FACTORIES,
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.live_model import ControllerProtocol, LiveModelEnv, SectionLiveField
from zcu_tools.gui.specs import make_waveform_spec_by_style

from .ports import ModuleLibraryWritePort
from .remote.path_resolver import (
    list_settable_paths,
    list_subtree_paths,
    resolve_and_set,
)

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


class _EditorCtrl(ControllerProtocol, Protocol):
    """Reactive-env surface a CfgEditor LiveModel needs.

    This is the LiveModel reactive environment (md/ml/device/bus reads) shared by
    every LiveField — see docs/adr/0007 (a legitimate narrow Port for "needs the
    owner's reactive env"). ML *writes* at commit time go through the separate
    ``ModuleLibraryWritePort``, not this protocol.
    """


class CfgEditorHost(_EditorCtrl, ModuleLibraryWritePort, Protocol):
    """Composition-root surface for ``CfgEditorService`` — the three facets it
    needs, all provided by the Controller: the reactive env (``_EditorCtrl``),
    the ML read/write port (``ModuleLibraryWritePort``), and the editor-version
    bump. The service itself decomposes this into the narrow dependencies; this
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

    Holds the ``SectionLiveField`` draft tree and owns the behaviour over it
    (set_field / get / validity / commit). Outside code reaches the draft only
    through this session (it is the gateway), and identifies it by ``editor_id``
    (DDD: reference aggregates by id). This replaces the old anemic dataclass
    whose behaviour lived entirely on the service (docs/adr/0008 §Aggregate Root).

    Lifecycle ownership (headless vs delegated) is the *Repository's* concern
    (``CfgEditorService``), not the aggregate's; the session only carries the
    discriminators the Repository needs.
    """

    editor_id: str
    root: SectionLiveField
    kind: str  # "headless" | "delegated"
    # headless: the ml entry kind being built ("module"/"waveform"); commit uses it.
    item_kind: Optional[str] = None
    # delegated: the owner's key (so close() can drop the owner→editor mapping).
    # A tab uses its tab_id; an inspect/writeback widget uses its own key.
    owner_key: Optional[str] = None
    # Monotonic open order; used for headless LRU eviction.
    seq: int = field(default=0)
    # on_change callback bound to root for the per-session change stream; held
    # so it can be disconnected when the session ends.
    change_cb: Optional[Callable[..., None]] = None

    # -- behaviour (the aggregate operates its own draft) ------------------

    def is_headless(self) -> bool:
        return self.kind == "headless"

    def current_paths(self) -> list[dict[str, object]]:
        """Full settable-path list of the draft (the ``get`` projection)."""
        return list_settable_paths(self.root)

    def set_field(self, path: str, value: object) -> dict[str, object]:
        """Mutate one field; return the changed sub-tree + draft validity."""
        resolve_and_set(self.root, path, _decode_value(value))
        return {
            "paths": list_subtree_paths(self.root, path),
            "valid": bool(self.root.is_valid()),
        }

    def lower(self, ml_port: ModuleLibraryWritePort) -> dict:
        """Lower the draft to a concrete raw dict (EvalValue → number)."""
        schema = CfgSchema(spec=self.root.spec, value=self.root.get_value())
        return schema_to_dict(schema, ml_port.get_current_ml())

    def commit(self, name: str, ml_port: ModuleLibraryWritePort) -> None:
        """Lower + register this headless draft into the ModuleLibrary.

        Registers via the write port; the Repository tears the session down only
        after this returns, so a validation failure leaves the draft intact.
        """
        if not self.is_headless():
            raise CfgEditorError(
                f"{self.editor_id!r} is a delegated session; commit applies only "
                "to headless ml-entry sessions"
            )
        raw = self.lower(ml_port)
        if self.item_kind == "module":
            ml_port.set_ml_module_from_raw(name, raw)
        else:
            ml_port.set_ml_waveform_from_raw(name, raw)


class CfgEditorService:
    """Repository for ``CfgEditorSession`` aggregates, keyed by a server id.

    Owns the session *lifecycle* (create / look up / reap) and the cross-session
    concerns (owner→id map, headless LRU, per-session change stream, version
    bump). Per-session editing behaviour lives on the aggregate
    (``CfgEditorSession``); the Repository delegates to it. The service holds no
    Qt objects.

    Dependencies (docs/adr/0008): ``env_ctrl`` is the LiveModel reactive env
    (narrow port); ``ml_port`` is the ModuleLibrary read/write port used at
    commit and to seed ``from_name`` sessions (no longer the whole Controller);
    ``version_bump`` / ``version_drop`` bump / forget the ``editor:<id>`` resource
    version (a registry-level concern since the id is Repository-assigned): bump on
    every edit (so commit's guard sees concurrent edits), drop on teardown (so a
    stale dependency on a gone session reads version 0).
    """

    def __init__(
        self,
        env_ctrl: "_EditorCtrl",
        ml_port: ModuleLibraryWritePort,
        version_bump: Callable[[str], None],
        version_drop: Callable[[str], None],
    ) -> None:
        self._env = LiveModelEnv(ctrl=env_ctrl)
        self._ml = ml_port
        self._version_bump = version_bump
        self._version_drop = version_drop
        self._editors: dict[str, CfgEditorSession] = {}
        self._owner_to_editor: dict[str, str] = {}
        self._seq = itertools.count()
        self._listener: Optional[ChangeListener] = None

    def set_change_listener(self, listener: Optional[ChangeListener]) -> None:
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
        discriminator: Optional[str] = None,
        from_name: Optional[str] = None,
    ) -> tuple[str, list[dict[str, object]]]:
        if item_kind not in _ITEM_KINDS:
            raise CfgEditorError(
                f"item_kind must be one of {_ITEM_KINDS}, got {item_kind!r}"
            )
        spec, value = self._initial_schema(item_kind, discriminator, from_name)
        root = SectionLiveField(spec, self._env, value)
        editor_id = self._new_id()
        session = CfgEditorSession(
            editor_id=editor_id,
            root=root,
            kind="headless",
            item_kind=item_kind,
            seq=next(self._seq),
        )
        self._editors[editor_id] = session
        self._attach_change_stream(session)
        self._evict_excess_headless()
        return editor_id, session.current_paths()

    def register_delegated_session(self, owner_key: str, root: SectionLiveField) -> str:
        """Register a widget's *existing* live LiveModel as a shared session.

        ``owner_key`` identifies the owner (a tab uses its tab_id; an inspect/
        writeback widget uses its own key). The root is the widget's own
        ``SectionLiveField`` — sharing the same instance is what makes a widget
        edit and an agent edit converge (WYSIWYG). A re-register for the same
        owner (e.g. inspect re-populates on type switch) closes the previous
        registration first. Lifetime is owner-driven: ``close`` drops the
        registration without tearing down the root.
        """
        prev = self._owner_to_editor.get(owner_key)
        if prev is not None:
            self.close(prev)
        editor_id = self._new_id()
        session = CfgEditorSession(
            editor_id=editor_id, root=root, kind="delegated", owner_key=owner_key
        )
        self._editors[editor_id] = session
        self._owner_to_editor[owner_key] = editor_id
        self._attach_change_stream(session)
        return editor_id

    def editor_id_for_owner(self, owner_key: str) -> Optional[str]:
        return self._owner_to_editor.get(owner_key)

    def set_field(self, editor_id: str, path: str, value: object) -> dict[str, object]:
        return self._require(editor_id).set_field(path, value)

    def get(self, editor_id: str) -> list[dict[str, object]]:
        return self._require(editor_id).current_paths()

    def commit(self, editor_id: str, name: str) -> None:
        # The aggregate lowers + registers itself through the ML write port; the
        # Repository only owns teardown (after a successful commit, so a
        # validation failure leaves the draft intact for the agent to fix).
        self._require(editor_id).commit(name, self._ml)
        self._remove(editor_id, teardown=True, reason="committed")

    def discard(self, editor_id: str) -> None:
        """Drop a *headless* session (tearing down its LiveModel).

        Delegated sessions are closed via ``close`` (which keeps the widget's
        root alive); discarding one would tear down a live widget tree.
        """
        session = self._require(editor_id)
        if not session.is_headless():
            raise CfgEditorError(
                f"{editor_id!r} is a delegated session; use close (it must not "
                "tear down the widget's live model)"
            )
        self._remove(editor_id, teardown=True, reason="discarded")

    def close(self, editor_id: str, *, reason: str = "tab_closed") -> None:
        """Remove a session registration without tearing down its LiveModel.

        Used for delegated sessions on owner teardown (tab close, dialog close):
        the widget's ``cfg_form.clear()`` owns teardown of the root, so the
        service must not double-tear-down. Unknown id is a no-op (close may race
        a never-registered owner).
        """
        session = self._editors.get(editor_id)
        if session is None:
            return
        self._remove(editor_id, teardown=False, reason=reason)

    def discard_for_client(self, editor_ids: list[str]) -> None:
        """Reclaim a batch of *headless* sessions (per-connection cleanup).

        Delegated sessions are owner-driven and never reclaimed on RPC
        disconnect, so they are skipped even if their id appears here. Unknown
        ids are ignored.
        """
        for editor_id in editor_ids:
            session = self._editors.get(editor_id)
            if session is not None and session.is_headless():
                self._remove(editor_id, teardown=True, reason="disconnected")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return "editor-" + uuid.uuid4().hex[:8]

    def _attach_change_stream(self, session: CfgEditorSession) -> None:
        """Bind a root ``on_change`` callback that pushes ``editor_changed``.

        Fires on any descendant edit (root on_change bubbles). The payload
        carries the full current path list (root on_change does not say which
        path changed); subtree-only optimisation can come later.
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
        session.root.on_change.connect(_on_change)

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
            session.root.on_change.disconnect(session.change_cb)
            session.change_cb = None
        if session.owner_key is not None:
            self._owner_to_editor.pop(session.owner_key, None)
        if teardown:
            session.root.teardown()
        # Forget the session's resource version (symmetric to tab/device drop):
        # a later stale dependency on this gone editor reads version 0 and the
        # guard treats it as stale, rather than spuriously matching a retained
        # version. Done whether or not we tear the root down — the session is gone.
        self._version_drop(editor_id)
        # Notify subscribers the session is gone (after state is consistent).
        self._emit(editor_id, "editor_closed", {"reason": reason})

    def _evict_excess_headless(self) -> None:
        headless = [
            (eid, s) for eid, s in self._editors.items() if s.kind == "headless"
        ]
        if len(headless) <= _MAX_HEADLESS_EDITORS:
            return
        # Evict oldest-first until back under the cap.
        headless.sort(key=lambda item: item[1].seq)
        for eid, _ in headless[: len(headless) - _MAX_HEADLESS_EDITORS]:
            self._remove(eid, teardown=True, reason="evicted")

    def _require(self, editor_id: str) -> CfgEditorSession:
        session = self._editors.get(editor_id)
        if session is None:
            raise CfgEditorError(f"unknown editor session: {editor_id!r}")
        return session

    def _initial_schema(
        self,
        item_kind: str,
        discriminator: Optional[str],
        from_name: Optional[str],
    ):
        if from_name is not None:
            ml = self._ml.get_current_ml()
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
    """Turn a tagged eval value into an ``EvalValue``; pass others through.

    The agent sends ``{"__kind": "eval", "expr": "..."}`` for an md-reference
    expression (the same tag used by the cfg-form codec). Everything else is a
    plain JSON scalar that ``resolve_and_set`` handles directly.
    """
    if isinstance(value, dict) and value.get("__kind") == "eval":
        expr = value.get("expr")
        if not isinstance(expr, str):
            raise CfgEditorError("eval value requires a string 'expr'")
        return EvalValue(expr=expr)
    return value
