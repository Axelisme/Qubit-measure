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
    """Controller surface a CfgEditor session needs.

    Extends the LiveModel env protocol with the ModuleLibrary registration
    entry points used at commit time (the same ones inspect_dialog calls).
    """

    def set_ml_module_from_raw(self, name: str, raw_dict: dict) -> None: ...
    def set_ml_waveform_from_raw(self, name: str, raw_dict: dict) -> None: ...

    # Bump the optimistic-concurrency version of an editor session's draft
    # resource (``editor:<id>``). The Controller forwards to State.version; the
    # session is the resource owner and calls this when its draft changes.
    def bump_editor_version(self, editor_id: str) -> None: ...


class CfgEditorError(RuntimeError):
    """A CfgEditor session operation failed (unknown id, bad kind, …)."""


@dataclass
class _EditorSession:
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


class CfgEditorService:
    """Owns headless LiveModel editing sessions keyed by a server-issued id.

    The controller passes itself as ``ctrl`` so sessions can read the live
    MetaDict / ModuleLibrary and build a ``LiveModelEnv``. The service holds no
    Qt objects; lifecycle is driven by the remote layer (per-connection) via
    ``discard_for_client``.
    """

    def __init__(self, ctrl: "_EditorCtrl") -> None:
        self._ctrl = ctrl
        self._env = LiveModelEnv(ctrl=ctrl)
        self._editors: dict[str, _EditorSession] = {}
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
        session = _EditorSession(
            root=root,
            kind="headless",
            item_kind=item_kind,
            seq=next(self._seq),
        )
        self._editors[editor_id] = session
        self._attach_change_stream(editor_id, session)
        self._evict_excess_headless()
        return editor_id, list_settable_paths(root)

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
        session = _EditorSession(root=root, kind="delegated", owner_key=owner_key)
        self._editors[editor_id] = session
        self._owner_to_editor[owner_key] = editor_id
        self._attach_change_stream(editor_id, session)
        return editor_id

    def editor_id_for_owner(self, owner_key: str) -> Optional[str]:
        return self._owner_to_editor.get(owner_key)

    def set_field(self, editor_id: str, path: str, value: object) -> dict[str, object]:
        session = self._require(editor_id)
        resolve_and_set(session.root, path, self._decode_value(value))
        return {
            "paths": list_subtree_paths(session.root, path),
            "valid": bool(session.root.is_valid()),
        }

    def get(self, editor_id: str) -> list[dict[str, object]]:
        session = self._require(editor_id)
        return list_settable_paths(session.root)

    def commit(self, editor_id: str, name: str) -> None:
        session = self._require(editor_id)
        if session.kind != "headless":
            raise CfgEditorError(
                f"{editor_id!r} is a delegated session; commit applies only to "
                "headless ml-entry sessions"
            )
        schema = CfgSchema(spec=session.root.spec, value=session.root.get_value())
        raw = schema_to_dict(schema, self._ctrl.get_current_ml())
        # Register first; only tear down the session once it lands, so a
        # validation failure leaves the draft intact for the agent to fix.
        if session.item_kind == "module":
            self._ctrl.set_ml_module_from_raw(name, raw)
        else:
            self._ctrl.set_ml_waveform_from_raw(name, raw)
        self._remove(editor_id, teardown=True, reason="committed")

    def discard(self, editor_id: str) -> None:
        """Drop a *headless* session (tearing down its LiveModel).

        Delegated sessions are closed via ``close`` (which keeps the widget's
        root alive); discarding one would tear down a live widget tree.
        """
        session = self._require(editor_id)
        if session.kind != "headless":
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
            if session is not None and session.kind == "headless":
                self._remove(editor_id, teardown=True, reason="disconnected")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return "editor-" + uuid.uuid4().hex[:8]

    def _attach_change_stream(self, editor_id: str, session: _EditorSession) -> None:
        """Bind a root ``on_change`` callback that pushes ``editor_changed``.

        Fires on any descendant edit (root on_change bubbles). The payload
        carries the full current path list (root on_change does not say which
        path changed); subtree-only optimisation can come later.
        """

        def _on_change(*_: object) -> None:
            # The draft for this session was just written (main thread); bump its
            # version so editor.commit's guard can detect a concurrent edit.
            self._ctrl.bump_editor_version(editor_id)
            self._emit(
                editor_id,
                "editor_changed",
                {"paths": list_settable_paths(session.root)},
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

    def _require(self, editor_id: str) -> _EditorSession:
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
            ml = self._ctrl.get_current_ml()
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

    @staticmethod
    def _decode_value(value: object) -> object:
        """Turn a tagged eval value into an ``EvalValue``; pass others through.

        The agent sends ``{"__kind": "eval", "expr": "..."}`` for an md-reference
        expression (the same tag used by the cfg-form codec). Everything else is
        a plain JSON scalar that ``resolve_and_set`` handles directly.
        """
        if isinstance(value, dict) and value.get("__kind") == "eval":
            expr = value.get("expr")
            if not isinstance(expr, str):
                raise CfgEditorError("eval value requires a string 'expr'")
            return EvalValue(expr=expr)
        return value
