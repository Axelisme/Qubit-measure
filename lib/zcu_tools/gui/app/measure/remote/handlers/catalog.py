"""Live catalog projection for the measure agent."""

# Handler names resolve through the wire registry's string references.
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.control_methods import CONTROL_METHOD_SPECS
from zcu_tools.gui.remote.param_spec import build_input_schema

from ..method_entries import METHOD_ENTRIES
from ..method_entries._registry import build_agent_catalog

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_rpc_catalog(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del adapter, params
    control_methods = [
        {
            "method": method,
            "description": spec.description,
            "params": build_input_schema(spec.params),
            "timeout_seconds": spec.timeout_seconds,
            "exposure": "rpc",
            "tool_names": [],
            "operation_key": None,
        }
        for method, spec in CONTROL_METHOD_SPECS.items()
    ]
    return {"methods": [*build_agent_catalog(METHOD_ENTRIES), *control_methods]}
