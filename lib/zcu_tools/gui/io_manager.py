from __future__ import annotations

import dataclasses
from typing import Optional

from zcu_tools.meta_tool import ExperimentManager

from .adapter import ExpContext


class IOManager:
    """Wraps ExperimentManager; returns new ExpContext objects to Controller."""

    def __init__(self) -> None:
        self._em: Optional[ExperimentManager] = None

    def setup(self, result_dir: str) -> None:
        self._em = ExperimentManager(result_dir)

    def list_contexts(self) -> list[str]:
        if self._em is None:
            return []
        return self._em.list_contexts()

    def use_context(self, label: str, base_ctx: ExpContext) -> ExpContext:
        """Switch to an existing context; preserve soc/soccfg/predictor/database_path."""
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        ml, md = self._em.use_flux(label)
        return dataclasses.replace(base_ctx, md=md, ml=ml, em=self._em)

    def new_context(
        self,
        base_ctx: ExpContext,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> ExpContext:
        """Create a new context; return updated ExpContext to Controller."""
        if self._em is None:
            raise RuntimeError("IOManager not set up. Call setup() first.")
        clone_src = (base_ctx.ml, base_ctx.md) if clone_from_current else None
        ml, md = self._em.new_flux(value=value, clone_from=clone_src, unit=unit)  # type: ignore[arg-type]
        # Flush files so list_contexts() and use_context() can find them immediately.
        md.dump()
        ml.dump()
        return dataclasses.replace(base_ctx, md=md, ml=ml, em=self._em)

    @property
    def has_project(self) -> bool:
        return self._em is not None

    def get_active_label(self) -> Optional[str]:
        if self._em is None:
            return None
        try:
            return self._em.label
        except RuntimeError:
            return None
