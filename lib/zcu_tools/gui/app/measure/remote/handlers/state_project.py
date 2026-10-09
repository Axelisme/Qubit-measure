"""State Project remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_state_has_project(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_project())}


def h_state_has_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_context())}


def h_state_has_active_context(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_active_context())}


def h_state_has_soc(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"value": bool(adapter.ctrl.has_soc())}


def h_state_hardware_gate(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {
        "active": [
            {
                "kind": presence.kind,
                "origin_kind": presence.origin_kind,
                "note": presence.note,
                "active_for_seconds": presence.active_for_seconds,
            }
            for presence in adapter.ctrl.get_hardware_gate_presence()
        ]
    }


def h_soc_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    include_cfg = bool(params["include_cfg"])  # ParamSpec(_bool_default)-validated
    return adapter.ctrl.get_soc_info(include_cfg=include_cfg)


def h_resources_versions(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"versions": adapter.ctrl.resources_versions()}
