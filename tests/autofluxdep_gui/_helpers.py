"""Shared test helpers — build ad-hoc providers without a real experiment.

The dependency-model tests need small providers with arbitrary declarations and
a scripted ``produce``. ``make_builder`` returns a ``Builder`` subclass instance
whose Node delegates to a supplied ``produce_fn(env, snapshot) -> Patch`` (or
returns an empty Patch). ``place`` wraps a Builder into a ``PlacedNode`` with
params. Together they replace the old ``NodeSpec`` + injected ``run_node``.
"""

from __future__ import annotations

from typing_extensions import Any, Callable, Optional
from zcu_tools.gui.app.autofluxdep.nodes.builder import (
    Builder,
    Node,
    PlacedNode,
    RunEnv,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep

ProduceFn = Callable[[RunEnv, Snapshot], Patch]


class _FnNode(Node):
    def __init__(self, env: RunEnv, fn: Optional[ProduceFn]) -> None:
        self._env = env
        self._fn = fn

    def produce(self, snapshot: Snapshot) -> Patch:
        if self._fn is None:
            return Patch()
        return self._fn(self._env, snapshot)


def make_builder(
    name: str,
    *,
    provides: tuple[str, ...] = (),
    requires: tuple[Dependency, ...] = (),
    optional: tuple[Dependency, ...] = (),
    requires_modules: tuple[ModuleDep, ...] = (),
    optional_modules: tuple[ModuleDep, ...] = (),
    provides_modules: tuple[str, ...] = (),
    base_params: tuple[str, ...] = (),
    produce_fn: Optional[ProduceFn] = None,
) -> Builder:
    """A Builder whose declarations are the given tuples and whose Node's
    ``produce`` calls ``produce_fn(env, snapshot)`` (or returns an empty Patch).
    """

    class _AdHocBuilder(Builder):
        def build_node(self, env: RunEnv) -> Node:
            return _FnNode(env, produce_fn)

    b = _AdHocBuilder()
    b.name = name
    b.provides = provides
    b.requires = requires
    b.optional = optional
    b.requires_modules = requires_modules
    b.optional_modules = optional_modules
    b.provides_modules = provides_modules
    b.base_params = base_params
    return b


def place(builder: Builder, **params: Any) -> PlacedNode:
    """Wrap ``builder`` into a PlacedNode with the given params."""
    return PlacedNode(builder=builder, params=dict(params))
