"""Shared session owners and fresh registry arrangement for writeback contracts."""

from dataclasses import dataclass

from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.resources.entry import ResultEntry

from zcu_lab.definitions import register_all


def writeback_registry() -> Registry:
    """Create and register a fresh adapter registry; propagate registration errors."""
    registry = Registry()
    register_all(registry)
    return registry


@dataclass
class WritebackSession:
    """Hold point control, context, service and entry for one fixture-owned session.

    The session fixture creates these owners with isolated temporary stores and
    restores component registration after use; consumers may edit its points.
    """

    control: ContextControlFacet
    context: ContextService
    service: WritebackService
    # Entry owning the independent points used to arrange reset scenarios.
    entry: ResultEntry
