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
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State

    from .ports import ProjectIOPort


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
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> None:
        logger.info(
            "new_context: value=%r unit=%r clone=%r", value, unit, clone_from_current
        )
        new_ctx = self._io.new_context(
            self._state.exp_context,
            value=value,
            unit=unit,
            clone_from_current=clone_from_current,
        )
        label = self._io.get_active_label() or ""
        new_ctx = dataclasses.replace(
            new_ctx, active_label=label, readiness=ContextReadiness.ACTIVE
        )
        self._state.set_context(new_ctx)
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def set_md_attr(self, key: str, value: Any) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        setattr(md, key, value)
        self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

    def del_md_attr(self, key: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        md = self._state.exp_context.md
        delattr(md, key)
        self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

    def set_ml_module(self, name: str, module: Any) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.register_module(**{name: module})
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

    def set_ml_module_from_raw(self, name: str, raw_dict: dict) -> None:
        """Deserialise raw dict into a Module cfg and register it under name.

        Raises MlEntryValidationError on validation / construction failure.
        """
        from zcu_tools.program.v2 import ModuleCfgFactory

        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        try:
            module = ModuleCfgFactory.from_raw(raw_dict, ml=ml)
        except Exception as exc:
            raise MlEntryValidationError(
                f"Invalid module configuration: {exc}"
            ) from exc
        ml.register_module(**{name: module})
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

    def del_ml_module(self, name: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.delete_module(name)
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

    def set_ml_waveform(self, name: str, waveform: Any) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.register_waveform(**{name: waveform})
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))

    def set_ml_waveform_from_raw(self, name: str, raw_dict: dict) -> None:
        """Deserialise raw dict into a Waveform cfg and register it under name.

        Raises MlEntryValidationError on validation / construction failure.
        """
        from zcu_tools.program.v2 import WaveformCfgFactory

        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        try:
            waveform = WaveformCfgFactory.from_raw(raw_dict, ml=ml)
        except Exception as exc:
            raise MlEntryValidationError(
                f"Invalid waveform configuration: {exc}"
            ) from exc
        ml.register_waveform(**{name: waveform})
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
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))
