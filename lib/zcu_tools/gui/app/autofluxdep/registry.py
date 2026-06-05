"""Builder registry — the experiment Node types the ``[+]`` button can add.

A flat name→Builder map of the available measurement experiment types. The UI
lists these in the add-Node menu; ``create_placement`` makes a fresh
``PlacedNode`` (Builder + empty params) for the chosen type.

Only *measurement* Builders are registered here — a Service (the predictor) is
NOT in this menu: it is loaded by the controller because a Node requires what it
provides, not placed by the user. Phase C registers ro_optimize / lenrabi / t1 /
t2ramsey / t2echo / mist alongside the qubit_freq worked example.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder

_BUILDERS: tuple[Builder, ...] = (QubitFreqBuilder(),)

NODE_BUILDERS: dict[str, Builder] = {b.name: b for b in _BUILDERS}


def available_node_types() -> list[str]:
    """Node type names the add-Node menu offers."""
    return list(NODE_BUILDERS)


def create_placement(type_name: str) -> PlacedNode:
    """A fresh ``PlacedNode`` of ``type_name`` with empty (default) params.

    Raises ``KeyError`` if the type is not registered (fast-fail).
    """
    return PlacedNode(builder=NODE_BUILDERS[type_name])
