from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Optional

from zcu_tools.gui.adapter import ContextReadiness
from zcu_tools.gui.event_bus import (
    ContextSwitchedPayload,
    GuiEvent,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, ExpContext
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State

    from .ports import ContextWrites, ProjectIOPort


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
        state: "State",
        io_manager: "ProjectIOPort",
        bus: "EventBus",
    ) -> None:
        self._state = state
        self._io = io_manager
        self._bus = bus

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

    def get_active_context_label(self) -> Optional[str]:
        return self._io.get_active_label()

    def get_context_labels(self) -> list[str]:
        return self._io.list_contexts()

    def get_current_md(self) -> MetaDict:
        return self._state.exp_context.md

    def get_current_ml(self) -> ModuleLibrary:
        return self._state.exp_context.ml

    def get_exp_context(self) -> "ExpContext":
        """The live ExpContext (md + ml + …) — used to seed role templates."""
        return self._state.exp_context

    def get_flux_dir(self) -> Optional[str]:
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
        new_ctx = dataclasses.replace(
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
        self._state.set_context(new_ctx)
        # md/ml content is fully swapped → bump context (path 3 of 3; see the
        # canonical anchor on ContextService.set_md_attr). set_context itself does
        # not bump, so context-switch callers bump here explicitly.
        self._state.version.bump("context")
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def use_context(self, label: str) -> None:
        logger.info("use_context: label=%r", label)
        new_ctx = self._io.use_context(label, self._state.exp_context)
        new_ctx = dataclasses.replace(
            new_ctx, active_label=label, readiness=ContextReadiness.ACTIVE
        )
        self._state.set_context(new_ctx)
        self._state.version.bump("context")
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "none",
        clone_from: Optional[str] = None,
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
        new_ctx = dataclasses.replace(
            new_ctx, active_label=label, readiness=ContextReadiness.ACTIVE
        )
        self._state.set_context(new_ctx)
        self._state.version.bump("context")
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def set_md_attr(self, key: str, value: Any) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        setattr(md, key, value)
        # Semantic context content change: bump so concurrency guards on
        # ``context`` (run.start / editor.commit / writeback.apply) detect this edit.
        #
        # CANONICAL ANCHOR — "writing md/ml must bump context" has TWO physical
        # paths (ADR-0011 collapsed writeback's direct write into path 1):
        #   1. ContextService writes: set_md_attr / del_md_attr / set_ml_*_from_schema /
        #      del_ml_* (field-level, each bumps+emits) and apply_writes (batch:
        #      one bump + one emit per kind). Writeback / editor commit / inspect /
        #      create_from_role all route here — the single write authority.
        #   2. context-switch: setup_project / use_context / new_context  (whole md/ml swap)
        # Both bump "context"; only set_context() itself does NOT (pure swap).
        self._state.version.bump("context")
        self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

    def del_md_attr(self, key: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        delattr(md, key)
        self._state.version.bump("context")
        self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

    # ------------------------------------------------------------------
    # ml/md content writes — the single write authority (ADR-0011).
    #
    # Sources holding an un-lowered CfgSchema (editor commit, writeback apply,
    # inspect save, create_from_role) write through set_ml_*_from_schema /
    # apply_writes; ContextService lowers (schema.to_raw_dict with the live md,
    # so callers can never forget md) + registers + bumps + emits. There is no
    # public raw-dict entry — raw lives only as an internal lowering detail.
    # ------------------------------------------------------------------

    def _lower_module(self, schema: "CfgSchema", ml: ModuleLibrary, md: MetaDict):
        from zcu_tools.program.v2 import ModuleCfgFactory

        raw = schema.to_raw_dict(md, ml)
        try:
            return ModuleCfgFactory.from_raw(raw, ml=ml)
        except Exception as exc:
            raise MlEntryValidationError(
                f"Invalid module configuration: {exc}"
            ) from exc

    def _lower_waveform(self, schema: "CfgSchema", ml: ModuleLibrary, md: MetaDict):
        from zcu_tools.program.v2 import WaveformCfgFactory

        raw = schema.to_raw_dict(md, ml)
        try:
            return WaveformCfgFactory.from_raw(raw, ml=ml)
        except Exception as exc:
            raise MlEntryValidationError(
                f"Invalid waveform configuration: {exc}"
            ) from exc

    def set_ml_module_from_schema(self, name: str, schema: "CfgSchema") -> None:
        """Lower (against live md) + register a Module cfg under name.

        Raises MlEntryValidationError on validation / construction failure.
        """
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ctx = self._state.exp_context
        module = self._lower_module(schema, ctx.ml, ctx.md)
        ctx.ml.register_module(**{name: module})
        self._state.version.bump("context")
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ctx.ml))

    def set_ml_waveform_from_schema(self, name: str, schema: "CfgSchema") -> None:
        """Lower (against live md) + register a Waveform cfg under name.

        Raises MlEntryValidationError on validation / construction failure.
        """
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ctx = self._state.exp_context
        waveform = self._lower_waveform(schema, ctx.ml, ctx.md)
        ctx.ml.register_waveform(**{name: waveform})
        self._state.version.bump("context")
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ctx.ml))

    def apply_writes(self, writes: "ContextWrites") -> None:
        """Apply a batch of md/ml content writes atomically (ADR-0011).

        One ``version.bump("context")``; at most one MD_CHANGED + one ML_CHANGED
        (the per-write methods would emit N times — a batch avoids N redundant
        full-refreshes). All lowering happens here, never at the call site.
        """
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ctx = self._state.exp_context
        for key, value in writes.md.items():
            setattr(ctx.md, key, value)
        for name, schema in writes.ml_modules.items():
            ctx.ml.register_module(**{name: self._lower_module(schema, ctx.ml, ctx.md)})
        for name, schema in writes.ml_waveforms.items():
            ctx.ml.register_waveform(
                **{name: self._lower_waveform(schema, ctx.ml, ctx.md)}
            )
        touched_ml = bool(writes.ml_modules or writes.ml_waveforms)
        if touched_ml and ctx.ml.has_persistence:
            ctx.ml.dump()
        if writes.md or touched_ml:
            self._state.version.bump("context")
        if writes.md:
            self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=ctx.md))
        if writes.ml_modules or writes.ml_waveforms:
            self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ctx.ml))

    def del_ml_module(self, name: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.delete_module(name)
        self._state.version.bump("context")
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

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
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

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
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

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
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))
