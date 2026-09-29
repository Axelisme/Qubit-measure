from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.expected_error import (
    ExpectedError,
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarValue,
    ValueInfo,
    ValueLookup,
    ValueRef,
    resolve_value_ref,
)
from zcu_tools.resources.context import MetaDict, ModuleLibrary
from zcu_tools.resources.context.content import (
    replace_context_contents,
    snapshot_context_contents,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.ports import ProjectIOPort
    from zcu_tools.gui.session.state import SessionState
    from zcu_tools.gui.session.types import SessionEnv


class MlEntryValidationError(InvalidInputError):
    """Expected failure when raw ML entry cannot be deserialised."""


class MdValueError(ValueError, ExpectedError):
    """Expected failure when the MetaDict value text cannot be coerced safely."""

    category = ExpectedErrorCategory.INVALID_INPUT


def _coerce_scalar(text: str, current: Any) -> Any:
    """Coerce text -> typed scalar.

    If `current` is None (key not yet present), accept int/float/bool/str only.
    Otherwise coerce to type(current); booleans must be 'true'/'false' (case
    insensitive). Numeric coercion uses the standard constructors and re-raises
    as MdValueError on failure.
    """
    stripped = text.strip()
    if current is None:
        # New key: try the most specific scalar first.
        if stripped.lower() in ("true", "false"):
            return stripped.lower() == "true"
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return float(stripped)
        except ValueError:
            pass
        return text  # raw string

    target_type = type(current)
    if target_type is bool:
        low = stripped.lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
        raise MdValueError(f"Expected bool (true/false) for existing key, got {text!r}")
    if target_type is int:
        try:
            return int(stripped)
        except ValueError as exc:
            raise MdValueError(f"Expected int, got {text!r}") from exc
    if target_type is float:
        try:
            return float(stripped)
        except ValueError as exc:
            raise MdValueError(f"Expected float, got {text!r}") from exc
    if target_type is str:
        return text
    raise MdValueError(
        f"Unsupported existing value type {target_type.__name__!r} for key — "
        "edit via structured tooling rather than the inline editor."
    )


def _validate_md_key(key: str) -> None:
    """Validate a user-facing MetaDict key before any content mutation."""
    if not isinstance(key, str):
        raise FailedPreconditionError(
            f"MetaDict keys must be str, got {type(key).__name__}"
        )
    if not key.strip():
        raise FailedPreconditionError("MetaDict key must not be empty.")
    try:
        MetaDict._ensure_data_key(key)
    except (AttributeError, TypeError) as exc:
        raise FailedPreconditionError(str(exc)) from exc


def _commit_md_snapshot(md: MetaDict, snapshot: Mapping[str, Any]) -> None:
    """Commit one already-validated MetaDict snapshot as a single mutation.

    MetaDict exposes attribute-level writes but no rename primitive. Replacing
    its data mapping under ContextService ownership avoids a caller-visible
    set-then-delete window and lets the service publish one version/event pair.
    The old in-memory mapping is restored if the underlying write fails.
    """
    previous = dict(md.items())
    previous_dirty = md._dirty
    try:
        md.require_writable()
        md.sync()
        md._data.clear()
        md._data.update(snapshot)
        md._dirty = True
        md.sync()
    except Exception:
        md._data.clear()
        md._data.update(previous)
        md._dirty = previous_dirty
        raise


class ContextService:
    """Encapsulates context switching, MetaDict/ModuleLibrary access, and project paths."""

    def __init__(
        self,
        state: SessionState,
        io_manager: ProjectIOPort,
        bus: BaseEventBus,
        values: ValueLookup | None = None,
    ) -> None:
        self._state = state
        self._io = io_manager
        self._bus = bus
        self._values = values or state.session_env.values
        if state.session_env.values is not self._values:
            # Pure facade injection: this preserves set_context's "no content bump"
            # semantics because md/ml are unchanged.
            self._state.set_context(self._attach_values(state.session_env))

    def _attach_values(self, ctx: SessionEnv) -> SessionEnv:
        if ctx.values is self._values:
            return ctx
        return dataclasses.replace(ctx, values=self._values)

    def has_project(self) -> bool:
        return self._io.has_project

    def has_context(self) -> bool:
        """True when any valid context exists (in-memory DRAFT or file-backed ACTIVE)."""
        return self._state.session_env.has_context()

    def has_draft_context(self) -> bool:
        return self._state.session_env.is_draft()

    def is_active_context(self) -> bool:
        """True only for a file-backed context eligible for run and save."""
        return self._state.session_env.is_active()

    def get_active_context_label(self) -> str | None:
        return self._io.get_active_label()

    def get_context_labels(self) -> list[str]:
        return self._io.list_contexts()

    def get_current_md(self) -> MetaDict:
        return self._state.session_env.md

    def get_current_ml(self) -> ModuleLibrary:
        return self._state.session_env.ml

    def get_session_env(self) -> SessionEnv:
        """The live SessionEnv (md + ml + …) — used to seed role templates."""
        return self._state.session_env

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._values.describe()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        ref = ValueRef(key=key, type_name=type_name)
        info = self._value_info(ref.key)
        return info, resolve_value_ref(ref, self._values)

    def _value_info(self, key: str) -> ValueInfo:
        for info in self._values.describe():
            if info.key == key:
                return info
        raise MissingValue(key, f"Value source {key!r} is not registered")

    def get_flux_dir(self) -> str | None:
        import os

        ctx = self._state.session_env
        label = self._io.get_active_label()
        if ctx.result_dir and label:
            return os.path.join(ctx.result_dir, "exps", label)
        return None

    def setup_project(self, result_dir: str) -> None:
        logger.info("setup_project: result_dir=%r", result_dir)
        self._io.setup(result_dir)

    def set_project_context(
        self,
        md: Any,
        ml: Any,
        chip_name: str = "unknown_chip",
        qub_name: str = "unknown_qubit",
        res_name: str = "unknown_resonator",
        result_dir: str = "",
        database_path: str = "",
    ) -> None:
        logger.info(
            "set_project_context: chip=%r qub=%r res=%r result_dir=%r db=%r",
            chip_name,
            qub_name,
            res_name,
            result_dir,
            database_path,
        )
        new_ctx = self._attach_values(
            dataclasses.replace(
                self._state.session_env,
                md=md,
                ml=ml,
                chip_name=chip_name,
                qub_name=qub_name,
                res_name=res_name,
                result_dir=result_dir,
                database_path=database_path,
                active_label="",
                readiness=ContextReadiness.DRAFT,
            )
        )
        self._state.set_context(new_ctx)
        # md/ml content is fully swapped → bump context (path 2 of 2; see the
        # canonical anchor on ContextService.set_md_attr). set_context itself does
        # not bump, so context-switch callers bump here explicitly.
        self._state.version.bump("context")
        self._bus.emit(
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def use_context(self, label: str) -> None:
        logger.info("use_context: label=%r", label)
        new_ctx = self._io.use_context(label, self._state.session_env)
        new_ctx = self._attach_values(
            dataclasses.replace(
                new_ctx, active_label=label, readiness=ContextReadiness.ACTIVE
            )
        )
        self._state.set_context(new_ctx)
        self._state.version.bump("context")
        self._bus.emit(
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def new_context(
        self,
        value: float | None = None,
        unit: str = "none",
        clone_from: str | None = None,
        label: str | None = None,
    ) -> None:
        if label is not None and (
            not label.strip()
            or label in {".", ".."}
            or any(
                char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in label
            )
        ):
            raise InvalidInputError(
                "context label must be a nonempty path segment without separators or control characters",
                reason_code="invalid_context_label",
            )
        if clone_from is not None:
            available = self._io.list_contexts()
            if clone_from not in available:
                raise InvalidInputError(
                    f"unknown context label: {clone_from!r}; available: {available}",
                    reason_code="unknown_context",
                )
        logger.info(
            "new_context: value=%r unit=%r clone_from=%r label=%r",
            value,
            unit,
            clone_from,
            label,
        )
        new_ctx = self._io.new_context(
            self._state.session_env,
            value=value,
            unit=unit,
            clone_from=clone_from,
            label=label,
        )
        label = self._io.get_active_label() or ""
        new_ctx = self._attach_values(
            dataclasses.replace(
                new_ctx, active_label=label, readiness=ContextReadiness.ACTIVE
            )
        )
        self._state.set_context(new_ctx)
        self._state.version.bump("context")
        self._bus.emit(
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def create_md_attr(self, key: str, value: Any) -> None:
        """Create one new MetaDict key after validating the complete request.

        Validation happens before the atomic snapshot commit, so collisions and
        invalid keys cannot leave a partially-created entry or emit a fact.
        """
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        _validate_md_key(key)
        md = self._state.session_env.md
        snapshot = dict(md.items())
        if key in snapshot:
            raise FailedPreconditionError(f"MetaDict already has attribute {key!r}.")
        snapshot[key] = value
        _commit_md_snapshot(md, snapshot)
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    def rename_md_attr(self, old: str, new: str) -> None:
        """Atomically rename one MetaDict key without a set-then-delete gap."""
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        _validate_md_key(old)
        _validate_md_key(new)
        md = self._state.session_env.md
        snapshot = dict(md.items())
        if old not in snapshot:
            raise FailedPreconditionError(f"MetaDict has no attribute {old!r}.")
        if new in snapshot:
            raise FailedPreconditionError(f"MetaDict already has attribute {new!r}.")
        snapshot[new] = snapshot.pop(old)
        _commit_md_snapshot(md, snapshot)
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    def set_md_attr(self, key: str, value: Any) -> None:
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        try:
            _validate_md_key(key)
        except FailedPreconditionError as exc:
            raise InvalidInputError(str(exc), reason_code="invalid_md_key") from exc
        md = self._state.session_env.md
        setattr(md, key, value)
        # Semantic context content change: bump so concurrency guards on
        # ``context`` (tab.run_start / editor.commit / tab.writeback_apply) detect this edit.
        #
        # CANONICAL ANCHOR — "a completed md/ml write bumps context" has TWO physical
        # paths (ADR-0067 collapsed writeback's direct write into path 1):
        #   1. ContextService writes: create_md_attr / rename_md_attr / set_md_attr /
        #      del_md_attr / replace_ml_*_from_schema / del_ml_* (field-level, each
        #      bumps+emits) and apply_ml_writes (batch: prepare without live writes, then one bump +
        #      one emit per kind; persistence errors occur after publication). Writeback / editor commit / inspect /
        #      create_from_role all route here — the single write authority.
        #   2. context-switch: setup_project / use_context / new_context  (whole md/ml swap)
        # Both bump "context"; only set_context() itself does NOT (pure swap).
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    def del_md_attr(self, key: str) -> None:
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        md = self._state.session_env.md
        try:
            delattr(md, key)
        except AttributeError as exc:
            raise FailedPreconditionError(str(exc)) from exc
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    # ------------------------------------------------------------------
    # ml/md content writes — the single write authority (ADR-0067).
    #
    # ``apply_ml_writes`` owns the *write sequence*: it sets md attrs +
    # registers the (lowered) ml entries and, once every step succeeds, bumps the
    # ``context`` version + emits at most one MD_CHANGED + one ML_CHANGED. It does
    # not roll back a partial failure. The CfgSchema *lowering* is
    # experiment-coupled, so it stays app-side and is injected as the
    # ``lower_module`` / ``lower_waveform`` callbacks (the app's ContextWritePort
    # façade builds them); this keeps ContextService free of the cfg-tree while
    # still being the sole owner of the bump/emit/persistence.
    # ------------------------------------------------------------------

    def apply_ml_writes(
        self,
        md: Mapping[str, Any],
        modules: Mapping[str, Any],
        waveforms: Mapping[str, Any],
        *,
        lower_module: Callable[[Any, ModuleLibrary, MetaDict], Any],
        lower_waveform: Callable[[Any, ModuleLibrary, MetaDict], Any],
        dump: bool,
    ) -> None:
        """Prepare a whole batch in memory, publish once, then persist.

        Later entries see earlier candidate writes, never partial live writes.
        Preparation failures leave both stores and their version untouched.
        Storage failures after publication report that settings were applied;
        they do not roll back or retry the batch. ``dump`` requests an explicit
        library dump in addition to the stores' normal synchronization policy.
        """
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        if not (md or modules or waveforms):
            return
        ctx = self._state.session_env
        candidate_md, candidate_ml = snapshot_context_contents(ctx.md, ctx.ml)
        candidate_md.update(md)
        for name, entry in modules.items():
            candidate_ml.register_module(
                **{name: lower_module(entry, candidate_ml, candidate_md)}
            )
        for name, entry in waveforms.items():
            candidate_ml.register_waveform(
                **{name: lower_waveform(entry, candidate_ml, candidate_md)}
            )
        touched_ml = bool(modules or waveforms)
        replace_context_contents(
            ctx.md,
            ctx.ml,
            metadata=candidate_md if md else None,
            library=candidate_ml if touched_ml else None,
        )
        self._state.version.bump("context")
        if md:
            self._bus.emit(MdChangedPayload(md=ctx.md))
        if touched_ml:
            self._bus.emit(MlChangedPayload(ml=ctx.ml))
        try:
            if md:
                ctx.md.sync()
            if touched_ml:
                if dump and ctx.ml.has_persistence:
                    ctx.ml.dump()
                else:
                    ctx.ml.sync()
        except Exception as exc:
            logger.exception("Context settings applied, but saving failed")
            raise RuntimeError("Settings were applied, but saving failed.") from exc

    def replace_ml_module_from_schema(
        self,
        old_name: str,
        new_name: str,
        schema: Any,
        *,
        lower_module: Callable[[Any, ModuleLibrary, MetaDict], Any],
        lower_waveform: Callable[[Any, ModuleLibrary, MetaDict], Any],
        dump: bool = False,
    ) -> None:
        """Atomically replace one module name and its cfg.

        The replacement validates the names and lowers the complete schema before
        touching the live ``ModuleLibrary``.  A collision, missing source, or
        lowering failure therefore leaves both the live entry and the caller's
        draft unchanged.  A successful replacement is one ContextService-owned
        content mutation: it bumps ``context`` once and emits one ``ML_CHANGED``.
        """
        self._replace_ml_entry_from_schema(
            "module",
            old_name,
            new_name,
            schema,
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=dump,
        )

    def replace_ml_waveform_from_schema(
        self,
        old_name: str,
        new_name: str,
        schema: Any,
        *,
        lower_module: Callable[[Any, ModuleLibrary, MetaDict], Any],
        lower_waveform: Callable[[Any, ModuleLibrary, MetaDict], Any],
        dump: bool = False,
    ) -> None:
        """Atomically replace one waveform name and its cfg; see module variant."""
        self._replace_ml_entry_from_schema(
            "waveform",
            old_name,
            new_name,
            schema,
            lower_module=lower_module,
            lower_waveform=lower_waveform,
            dump=dump,
        )

    def _replace_ml_entry_from_schema(
        self,
        item_kind: str,
        old_name: str,
        new_name: str,
        schema: Any,
        *,
        lower_module: Callable[[Any, ModuleLibrary, MetaDict], Any],
        lower_waveform: Callable[[Any, ModuleLibrary, MetaDict], Any],
        dump: bool = False,
    ) -> None:
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        if not old_name or not new_name:
            raise FailedPreconditionError("ModuleLibrary names must not be empty.")

        ctx = self._state.session_env
        store = ctx.ml.modules if item_kind == "module" else ctx.ml.waveforms
        if old_name not in store:
            raise FailedPreconditionError(f"No {item_kind} named {old_name!r}.")
        if old_name != new_name and new_name in store:
            raise FailedPreconditionError(
                f"A {item_kind} named {new_name!r} already exists."
            )

        # Lower and construct the replacement before the first live write.  The
        # lower callback is the app-owned cfg boundary; ContextService remains
        # the only owner of the resulting ModuleLibrary mutation.
        lower = lower_module if item_kind == "module" else lower_waveform
        replacement = lower(schema, ctx.ml, ctx.md)
        if item_kind == "module":
            ctx.ml.register_module(**{new_name: replacement})
            if old_name != new_name:
                ctx.ml.delete_module(old_name)
        else:
            ctx.ml.register_waveform(**{new_name: replacement})
            if old_name != new_name:
                ctx.ml.delete_waveform(old_name)

        if dump and ctx.ml.has_persistence:
            ctx.ml.dump()
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ctx.ml))

    def del_ml_module(self, name: str) -> None:
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        ml = self._state.session_env.ml
        ml.delete_module(name)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))

    def coerce_md_value(self, key: str, text: str) -> Any:
        """Convert a user-typed string into a typed value for MetaDict[key].

        - If the key exists in the live MetaDict, coerce to that key's existing
          Python type (int/float/bool/str) and reject conversions that lose
          information.
        - If the key does not yet exist, accept only scalar text values: int,
          float, bool, and bare strings. Reject complex literals (lists, tuples,
          dicts) — those need to go through a structured editor, not a string
          parser. This is intentionally narrower than ast.literal_eval, which
          turned `"1, 2"` into a tuple silently.

        Raises MdValueError on any conversion that cannot be performed safely.
        """
        existing = self._state.session_env.md
        current = getattr(existing, key, None) if self.has_context() else None
        return _coerce_scalar(text, current)

    def del_ml_waveform(self, name: str) -> None:
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        ml = self._state.session_env.ml
        ml.delete_waveform(name)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))

    def rename_ml_module(self, old: str, new: str) -> None:
        """Rename an ml module by re-registering under ``new`` and deleting ``old``.

        LINKED cfg references keep ``old`` and become invalid while it is missing;
        restoring the key relinks them. MODIFIED references keep their inline
        Custom values. The single ML_CHANGED below refreshes drafts. A clash
        at the new name fails fast.
        """
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        if not new:
            raise FailedPreconditionError("New name must not be empty.")
        ml = self._state.session_env.ml
        if old not in ml.modules:
            raise FailedPreconditionError(f"No module named {old!r}.")
        if new in ml.modules:
            raise FailedPreconditionError(f"A module named {new!r} already exists.")
        ml.register_module(**{new: ml.modules[old]})
        ml.delete_module(old)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))

    def rename_ml_waveform(self, old: str, new: str) -> None:
        """Rename an ml waveform (see :meth:`rename_ml_module`)."""
        if not self.has_context():
            raise FailedPreconditionError("No experiment context.")
        if not new:
            raise FailedPreconditionError("New name must not be empty.")
        ml = self._state.session_env.ml
        if old not in ml.waveforms:
            raise FailedPreconditionError(f"No waveform named {old!r}.")
        if new in ml.waveforms:
            raise FailedPreconditionError(f"A waveform named {new!r} already exists.")
        ml.register_waveform(**{new: ml.waveforms[old]})
        ml.delete_waveform(old)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))
