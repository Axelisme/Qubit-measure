"""Live catalog projection for the measure agent."""

# Handler names resolve through the wire registry's string references.
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..method_entries import METHOD_ENTRIES
from ..method_entries._registry import build_agent_catalog

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_rpc_catalog(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del adapter, params
    return {"methods": build_agent_catalog(METHOD_ENTRIES)}
