from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Optional

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
    from zcu_tools.gui.io_manager import IOManager
    from zcu_tools.gui.state import State


class ContextService:
    """Encapsulates context switching, MetaDict/ModuleLibrary access, and project paths."""

    def __init__(
        self,
        state: "State",
        io_manager: "IOManager",
        bus: "EventBus",
    ) -> None:
        self._state = state
        self._io = io_manager
        self._bus = bus

    def has_project(self) -> bool:
        return self._io.has_project

    def has_context(self) -> bool:
        """True when any valid context exists (startup empty ctx or file-backed flux ctx)."""
        return self._io.has_context or self._state.has_startup_context

    def has_startup_context(self) -> bool:
        return self._state.has_startup_context

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
        )
        self._state.set_context(new_ctx)
        self._state.has_startup_context = True
        self._bus.emit(
            GuiEvent.CONTEXT_SWITCHED,
            ContextSwitchedPayload(md=new_ctx.md, ml=new_ctx.ml),
        )

    def use_context(self, label: str) -> None:
        logger.info("use_context: label=%r", label)
        new_ctx = self._io.use_context(label, self._state.exp_context)
        new_ctx = dataclasses.replace(new_ctx, active_label=label)
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
        new_ctx = dataclasses.replace(new_ctx, active_label=label)
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

    def del_ml_waveform(self, name: str) -> None:
        if not self.has_context():
            raise RuntimeError("No experiment context.")
        ml = self._state.exp_context.ml
        ml.delete_waveform(name)
        self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ml))
