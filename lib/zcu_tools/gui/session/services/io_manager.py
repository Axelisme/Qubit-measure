from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from zcu_tools.gui.session.types import ExpContext

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ExperimentManager


class IOManager:
    """Wraps ExperimentManager; returns new ExpContext objects to Controller."""

    def __init__(self) -> None:
        self._em: ExperimentManager | None = None

    def setup(self, result_dir: str) -> None:
        from zcu_tools.meta_tool import ExperimentManager

        logger.info("setup: result_dir=%r", result_dir)
        self._em = ExperimentManager(result_dir)

    def list_contexts(self) -> list[str]:
        if self._em is None:
            return []
        return self._em.list_contexts()

    def use_context(self, label: str, base_ctx: ExpContext) -> ExpContext:
        """Switch to an existing context; preserve soc/soccfg/predictor/database_path."""
        logger.info("use_context: label=%r", label)
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        ml, md = self._em.use_flux(label)
        return dataclasses.replace(base_ctx, md=md, ml=ml)

    def new_context(
        self,
        base_ctx: ExpContext,
        value: float | None = None,
        unit: str = "none",
        clone_from: str | None = None,
    ) -> ExpContext:
        """Create a new context; return updated ExpContext to Controller.

        ``clone_from`` is the label of an existing context to clone (its ml/md
        are read from ``exp_dir/<label>``); ``None`` starts empty. ``em.new_flux``
        already accepts a label string as ``clone_from``.
        """
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        ml, md = self._em.new_flux(value=value, clone_from=clone_from, unit=unit)  # type: ignore[arg-type]
        # Flush files so list_contexts() and use_context() can find them immediately.
        md.dump()
        ml.dump()
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
