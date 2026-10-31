"""Adapt core callables to explicit operation capabilities."""

from zcu_tools.cfg_model import ConfigBase

from .operations import Completed, RunFunction
from .records import RunRecord
from .scope import Scope


async def run_core[Cfg: ConfigBase, Result](
    scope: Scope,
    experiment: RunFunction[Cfg, Result],
    cfg: Cfg,
    *,
    name: str | None = None,
) -> Completed[RunRecord[Cfg, Result]]:
    """Run a core once with an owned cfg and operation resources, without saving."""
    raise NotImplementedError("Core adaptation and pre-call metadata are not implemented")
