from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.gui.session.value_lookup import ValueLookup
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.ports import ProjectIOPort
    from zcu_tools.gui.session.state import SessionState
    from zcu_tools.gui.session.types import ExpContext


class MlEntryValidationError(RuntimeError):
    """Expected failure when raw ML entry cannot be deserialised."""


class MdValueError(ValueError):
    """Expected failure when the MetaDict value text cannot be coerced safely."""


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
        self._values = values or state.exp_context.values
        if state.exp_context.values is not self._values:
            # Pure facade injection: this preserves set_context's "no content bump"
            # semantics because md/ml are unchanged.
            self._state.set_context(self._attach_values(state.exp_context))

    def _attach_values(self, ctx: ExpContext) -> ExpContext:
        if ctx.values is self._values:
            return ctx
        return dataclasses.replace(ctx, values=self._values)

    def has_project(self) -> bool:
        return self._io.has_project

    def has_context(self) -> bool:
        """True when any valid context exists (startup DRAFT or file-backed ACTIVE)."""
        return self._state.exp_context.has_context()

    def has_startup_context(self) -> bool:
        return self._state.exp_context.is_draft()

    def is_active_context(self) -> bool:
        """True only for a file-backed context eligible for run and save."""
        return self._state.exp_context.is_active()

    def get_active_context_label(self) -> str | None:
        return self._io.get_active_label()

    def get_context_labels(self) -> list[str]:
        return self._io.list_contexts()

    def get_current_md(self) -> MetaDict:
        return self._state.exp_context.md

    def get_current_ml(self) -> ModuleLibrary:
        return self._state.exp_context.ml

    def get_exp_context(self) -> ExpContext:
        """The live ExpContext (md + ml + …) — used to seed role templates."""
        return self._state.exp_context

    def get_flux_dir(self) -> str | None:
        import os

        ctx = self._state.exp_context
        label = self._io.get_active_label()
        if ctx.result_dir and label:
            return os.path.join(ctx.result_dir, "exps", label)
        return None

    def setup_project(self, result_dir: str) -> None:
        logger.info("setup_project: result_dir=%r", result_dir)
        self._io.setup(result_dir)

    def set_startup_context(
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
            "set_startup_context: chip=%r qub=%r res=%r result_dir=%r db=%r",
            chip_name,
            qub_name,
            res_name,
            result_dir,
            database_path,
        )
        new_ctx = self._attach_values(
            dataclasses.replace(
                self._state.exp_context,
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
        # md/ml content is fully swapped → bump context (path 3 of 3; see the
        # canonical anchor on ContextService.set_md_attr). set_context itself does
        # not bump, so context-switch callers bump here explicitly.
        self._state.version.bump("context")
        self._bus.emit(
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def use_context(self, label: str) -> None:
        logger.info("use_context: label=%r", label)
        new_ctx = self._io.use_context(label, self._state.exp_context)
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
    ) -> None:
        logger.info(
            "new_context: value=%r unit=%r clone_from=%r", value, unit, clone_from
        )
        new_ctx = self._io.new_context(
            self._state.exp_context,
            value=value,
            unit=unit,
            clone_from=clone_from,
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

    def set_md_attr(self, key: str, value: Any) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        setattr(md, key, value)
        # Semantic context content change: bump so concurrency guards on
        # ``context`` (tab.run_start / editor.commit / tab.writeback_apply) detect this edit.
        #
        # CANONICAL ANCHOR — "writing md/ml must bump context" has TWO physical
        # paths (ADR-0006 collapsed writeback's direct write into path 1):
        #   1. ContextService writes: set_md_attr / del_md_attr / set_ml_*_from_schema /
        #      del_ml_* (field-level, each bumps+emits) and apply_writes (batch:
        #      one bump + one emit per kind). Writeback / editor commit / inspect /
        #      create_from_role all route here — the single write authority.
        #   2. context-switch: setup_project / use_context / new_context  (whole md/ml swap)
        # Both bump "context"; only set_context() itself does NOT (pure swap).
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    def del_md_attr(self, key: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        delattr(md, key)
        self._state.version.bump("context")
        self._bus.emit(MdChangedPayload(md=md))

    # ------------------------------------------------------------------
    # ml/md content writes — the single write authority (ADR-0006).
    #
    # ``apply_ml_writes`` owns the *write transaction*: it sets md attrs +
    # registers the (lowered) ml entries, then bumps the ``context`` version +
    # emits at most one MD_CHANGED + one ML_CHANGED. The CfgSchema *lowering* is
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
        """Apply a batch of md/ml content writes atomically (ADR-0006).

        ``md`` maps attr → value; ``modules`` / ``waveforms`` map entry name → an
        opaque un-lowered entry (a ``CfgSchema``), lowered here via the injected
        ``lower_module`` / ``lower_waveform`` (app-side; the cfg-tree never enters
        this module). Lowering is interleaved with registration so a later entry
        sees an earlier one. One ``version.bump("context")`` and at most one
        MD_CHANGED + one ML_CHANGED (a batch avoids N redundant full-refreshes).
        ``dump`` persists the ml when it has persistence (writeback batch persists;
        a single editor commit does not). Raises MlEntryValidationError (from the
        lowering callback) on a bad entry."""
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ctx = self._state.exp_context
        for key, value in md.items():
            setattr(ctx.md, key, value)
        for name, entry in modules.items():
            ctx.ml.register_module(**{name: lower_module(entry, ctx.ml, ctx.md)})
        for name, entry in waveforms.items():
            ctx.ml.register_waveform(**{name: lower_waveform(entry, ctx.ml, ctx.md)})
        touched_ml = bool(modules or waveforms)
        if dump and touched_ml and ctx.ml.has_persistence:
            ctx.ml.dump()
        if md or touched_ml:
            self._state.version.bump("context")
        if md:
            self._bus.emit(MdChangedPayload(md=ctx.md))
        if modules or waveforms:
            self._bus.emit(MlChangedPayload(ml=ctx.ml))

    def del_ml_module(self, name: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
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
        existing = self._state.exp_context.md
        current = getattr(existing, key, None) if self.has_context() else None
        return _coerce_scalar(text, current)

    def del_ml_waveform(self, name: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.delete_waveform(name)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))

    def rename_ml_module(self, old: str, new: str) -> None:
        """Rename an ml module by re-registering under ``new`` and deleting ``old``.

        References (cfg ``chosen_key == old``) are NOT migrated — they degrade
        to inline Custom via the ModuleRefLiveField self-heal on the single
        ML_CHANGED below (the value is preserved). New-name clash fails fast.
        """
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        if not new:
            raise RuntimeError("New name must not be empty.")
        ml = self._state.exp_context.ml
        if old not in ml.modules:
            raise RuntimeError(f"No module named {old!r}.")
        if new in ml.modules:
            raise RuntimeError(f"A module named {new!r} already exists.")
        ml.register_module(**{new: ml.modules[old]})
        ml.delete_module(old)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))

    def rename_ml_waveform(self, old: str, new: str) -> None:
        """Rename an ml waveform (see :meth:`rename_ml_module`)."""
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        if not new:
            raise RuntimeError("New name must not be empty.")
        ml = self._state.exp_context.ml
        if old not in ml.waveforms:
            raise RuntimeError(f"No waveform named {old!r}.")
        if new in ml.waveforms:
            raise RuntimeError(f"A waveform named {new!r} already exists.")
        ml.register_waveform(**{new: ml.waveforms[old]})
        ml.delete_waveform(old)
        self._state.version.bump("context")
        self._bus.emit(MlChangedPayload(ml=ml))
