"""Live agent catalog wire method."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "rpc.catalog",
        "catalog:h_rpc_catalog",
        MethodSpec(5.0, "Read the live agent-facing method catalog."),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
)
