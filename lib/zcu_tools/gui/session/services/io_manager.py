from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from zcu_tools.gui.session.types import SessionEnv

if TYPE_CHECKING:
    from zcu_tools.resources.context import ContextManager


class IOManager:
    """Wraps ContextManager; returns new SessionEnv objects to Controller."""

    def __init__(self) -> None:
        self._em: ContextManager | None = None

    def setup(self, result_dir: str) -> None:
        from zcu_tools.resources.context import ContextManager

        logger.info("setup: result_dir=%r", result_dir)
        self._em = ContextManager(Path(result_dir) / "exps")

    def list_contexts(self) -> list[str]:
        if self._em is None:
            return []
        return self._em.list_contexts()

    def use_context(self, label: str, base_ctx: SessionEnv) -> SessionEnv:
        """Switch to an existing context; preserve soc/soccfg/predictor/database_path."""
        logger.info("use_context: label=%r", label)
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        ml, md = self._em.use_flux(label)
        return dataclasses.replace(base_ctx, md=md, ml=ml)

    def new_context(
        self,
        base_ctx: SessionEnv,
        value: float | None = None,
        unit: str = "none",
        clone_from: str | None = None,
        label: str | None = None,
    ) -> SessionEnv:
        """Create a new context; return updated SessionEnv to Controller.

        ``clone_from`` is the label of an existing context to clone (its ml/md
        are read from ``exp_dir/<label>``); ``None`` starts empty. ``em.new_flux``
        already accepts a label string as ``clone_from``.
        """
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        if unit not in ("A", "V", "K", "none"):
            raise ValueError(f"unsupported context unit: {unit!r}")
        ml, md = self._em.new_flux(
            value=value, clone_from=clone_from, label=label, unit=unit
        )
        return dataclasses.replace(base_ctx, md=md, ml=ml)

    @property
    def has_project(self) -> bool:
        return self._em is not None

    @property
    def has_context(self) -> bool:
        """True only when a flux context (md/ml) has been selected."""
        return self._em is not None and self._em.current_label is not None

    def get_active_label(self) -> str | None:
        if self._em is None:
            return None
        return self._em.current_label
