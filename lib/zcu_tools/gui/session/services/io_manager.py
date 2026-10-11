from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.resources.context import ContextManager

logger = logging.getLogger(__name__)


class IOManager:
    """Manage the project context root, entries, and active-context state."""

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
