"""CfgEditorService — headless, stateful editing of ModuleLibrary entries.

This is the RPC-side owner of a LiveModel draft, the symmetric counterpart to
the View-side ``inspect_dialog`` (which owns a Qt-coupled ``CfgFormWidget``
draft). Both build on the same Qt-free machinery — ``SectionLiveField``,
``resolve_and_set`` / ``list_settable_paths`` and ``schema_to_dict`` — but each
manages its own session lifecycle: the dialog binds to a Qt dialog open/close,
this service binds to an RPC connection (see ``_ClientState.editor_ids``).

A session lets an MCP agent build/edit a module or waveform incrementally:

    open(item_kind, discriminator | from_name) -> (editor_id, paths)
    set_field(editor_id, path, value)          -> (paths, valid)
    get(editor_id)                             -> paths
    commit(editor_id, name)                    -> {}     (lowers, registers)
    discard(editor_id)                         -> {}

The incremental shape is *required*, not a convenience: ModuleRef/WaveformRef
key switches rebuild the field sub-tree, so the agent cannot send one complete
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

import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

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

_ITEM_KINDS = ("module", "waveform")


class _EditorCtrl(ControllerProtocol, Protocol):
    """Controller surface a CfgEditor session needs.

    Extends the LiveModel env protocol with the ModuleLibrary registration
    entry points used at commit time (the same ones inspect_dialog calls).
    """

    def set_ml_module_from_raw(self, name: str, raw_dict: dict) -> None: ...
    def set_ml_waveform_from_raw(self, name: str, raw_dict: dict) -> None: ...


class CfgEditorError(RuntimeError):
    """A CfgEditor session operation failed (unknown id, bad kind, …)."""


@dataclass
class _EditorSession:
    item_kind: str
    root: SectionLiveField


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
        editor_id = "editor-" + uuid.uuid4().hex[:8]
        self._editors[editor_id] = _EditorSession(item_kind=item_kind, root=root)
        return editor_id, list_settable_paths(root)

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
        schema = CfgSchema(spec=session.root.spec, value=session.root.get_value())
        raw = schema_to_dict(schema, self._ctrl.get_current_ml())
        # Register first; only tear down the session once it lands, so a
        # validation failure leaves the draft intact for the agent to fix.
        if session.item_kind == "module":
            self._ctrl.set_ml_module_from_raw(name, raw)
        else:
            self._ctrl.set_ml_waveform_from_raw(name, raw)
        session.root.teardown()
        del self._editors[editor_id]

    def discard(self, editor_id: str) -> None:
        session = self._require(editor_id)
        session.root.teardown()
        del self._editors[editor_id]

    def discard_for_client(self, editor_ids: list[str]) -> None:
        """Tear down a batch of sessions (per-connection cleanup); ignore unknown."""
        for editor_id in editor_ids:
            session = self._editors.pop(editor_id, None)
            if session is not None:
                session.root.teardown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
