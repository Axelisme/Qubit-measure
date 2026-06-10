"""Plot routing — task-local destination for matplotlib output.

One of the shared plotting-substrate modules (see also ``backend`` =
interception client, ``host`` = Qt canvas lifecycle + main-thread bridge).

Behaviour guarantee this module provides:
- The active ``FigureContainer`` is held in a ``ContextVar`` (task-local), NOT a
  global. Each run/analyze worker sets its own routing scope at entry via
  ``routing_scope(container)``; concurrent workers therefore never cross-route
  their figures even though ``mpl_backend`` reads a single "current container".
- ``ContextVar`` semantics mean a child context (a QThread that inherited the
  parent snapshot) sees the value bound *at copy time*; a worker MUST enter its
  own ``routing_scope`` rather than rely on inheritance. Outside any scope the
  current container is ``None`` (backend falls back to a detached figure).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from .container import FigureContainer

_current_container: ContextVar[Optional["FigureContainer"]] = ContextVar(
    "zcu_tools_gui_current_figure_container",
    default=None,
)

RoutingToken = Token[Optional["FigureContainer"]]


@dataclass(frozen=True)
class RoutingStateSnapshot:
    has_current_container: bool
    current_container_id: Optional[int]


def set_current_container(container: "FigureContainer") -> RoutingToken:
    return _current_container.set(container)


def reset_current_container(token: RoutingToken) -> None:
    _current_container.reset(token)


def get_current_container() -> Optional["FigureContainer"]:
    return _current_container.get()


def require_current_container() -> "FigureContainer":
    container = get_current_container()
    if container is None:
        raise RuntimeError("No active FigureContainer")
    return container


def has_current_container() -> bool:
    return get_current_container() is not None


@contextmanager
def routing_scope(container: Optional["FigureContainer"]) -> Iterator[None]:
    if container is None:
        yield
        return

    token = set_current_container(container)
    try:
        yield
    finally:
        reset_current_container(token)


def dump_routing_state() -> RoutingStateSnapshot:
    container = get_current_container()
    return RoutingStateSnapshot(
        has_current_container=container is not None,
        current_container_id=id(container) if container is not None else None,
    )


__all__ = [
    "RoutingStateSnapshot",
    "RoutingToken",
    "dump_routing_state",
    "get_current_container",
    "has_current_container",
    "require_current_container",
    "reset_current_container",
    "routing_scope",
    "set_current_container",
]
