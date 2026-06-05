"""Node type registry — the NodeSpecs the ``[+]`` button can add.

A flat name→NodeSpec map of the available experiment Node types. The UI lists
these in the add-Node menu; ``create_instance`` makes a fresh NodeInstance with
default params. Phase C registers lenrabi / t1 / t2ramsey / t2echo / mist
alongside the qubit_freq / ro_optimize worked examples.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QUBIT_FREQ_SPEC
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RO_OPTIMIZE_SPEC
from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance, NodeSpec

NODE_SPECS: dict[str, NodeSpec] = {
    spec.name: spec for spec in (QUBIT_FREQ_SPEC, RO_OPTIMIZE_SPEC)
}


def available_node_types() -> list[str]:
    """Node type names the add-Node menu offers."""
    return list(NODE_SPECS)


def create_instance(type_name: str) -> NodeInstance:
    """A fresh NodeInstance of ``type_name`` with empty (default) params.

    Raises ``KeyError`` if the type is not registered (fast-fail).
    """
    return NodeInstance(spec=NODE_SPECS[type_name])
