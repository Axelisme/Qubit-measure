"""Builder registry — the experiment Node types the ``[+]`` button can add.

A flat name→Builder map of the available measurement experiment types. The UI
lists these in the add-Node menu; ``create_placement`` makes a fresh
``PlacedNode`` (Builder + empty params) for the chosen type.

Only *measurement* Builders are registered here — a Service (the predictor) is
NOT in this menu: it is loaded by the controller because a Node requires what it
provides, not placed by the user.

The registered Builders form a dependency chain (resolved latest-available, no
topo sort): qubit_freq → lenrabi (needs qubit_freq, produces pi/pi2 pulses) →
ro_optimize / t1 (need pi_pulse) → t2ramsey / t2echo (need pi/pi2 pulses); mist
needs pi_pulse. The user orders them in the list; missing deps skip a Node for
that flux point.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder

_BUILDERS: tuple[Builder, ...] = (
    QubitFreqBuilder(),
    LenRabiBuilder(),
    RoOptimizeBuilder(),
    T1Builder(),
    T2RamseyBuilder(),
    T2EchoBuilder(),
    MistBuilder(),
)

NODE_BUILDERS: dict[str, Builder] = {b.name: b for b in _BUILDERS}


def available_node_types() -> list[str]:
    """Node type names the add-Node menu offers."""
    return list(NODE_BUILDERS)


def create_placement(type_name: str) -> PlacedNode:
    """A fresh ``PlacedNode`` of ``type_name`` with empty (default) params.

    Raises ``KeyError`` if the type is not registered (fast-fail).
    """
    return PlacedNode(builder=NODE_BUILDERS[type_name])
