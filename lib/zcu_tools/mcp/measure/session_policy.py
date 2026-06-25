"""Measure-gui MCP policy tables and pure helpers.

The shared :class:`~zcu_tools.mcp.core.bridge.McpBridge` is transport-only
(ADR-0014).  These tables are measure-gui app policy and are executed by
``MeasureMcpSession``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

VersionPatternMap = Mapping[str, tuple[str, ...]]
OperationKeyFn = Callable[[dict[str, Any]], str]


_GUARD_DEPS_DATA: dict[str, tuple[str, ...]] = {
    # ``device:*`` guards mutations of existing devices; ``devices:__set__``
    # guards membership because a glob cannot reveal devices added later.
    "tab.run_start": (
        "tab:{tab_id}:cfg",
        "tab:{tab_id}",
        "soc",
        "context",
        "device:*",
        "devices:__set__",
    ),
    "tab.load_data": (
        "tab:{tab_id}",
        "tab:{tab_id}:result",
        "tab:{tab_id}:analyze",
        "context",
    ),
    "tab.save_data": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "tab.save_image": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "tab.save_post_image": ("tab:{tab_id}:post_analyze", "tab:{tab_id}:save_path"),
    "tab.save_result": ("tab:{tab_id}:result", "tab:{tab_id}:save_path"),
    "tab.save_set_paths": ("tab:{tab_id}:save_path",),
    "tab.writeback_set": ("tab:{tab_id}:result", "tab:{tab_id}:analyze", "context"),
    "tab.writeback_apply": (
        "tab:{tab_id}:result",
        "tab:{tab_id}:analyze",
        "context",
    ),
    "editor.commit": ("editor:{editor_id}", "context"),
    "arb_waveform.set": ("arb_waveforms",),
}

GUARD_DEPS: VersionPatternMap = MappingProxyType(_GUARD_DEPS_DATA)


_READ_REVEALS_DATA: dict[str, tuple[str, ...]] = {
    "tab.get_cfg": ("tab:{tab_id}:cfg",),
    "editor.get": ("editor:{editor_id}",),
    "device.snapshot": ("device:{name}",),
    "device.list": ("devices:__set__",),
    "arb_waveform.list": ("arb_waveforms",),
    "arb_waveform.preview": ("arb_waveforms",),
}

READ_REVEALS: VersionPatternMap = MappingProxyType(_READ_REVEALS_DATA)


_OPERATION_KEY_OF_DATA: dict[str, OperationKeyFn] = {
    "device.connect": lambda p: f"device:{p.get('name', '')}",
    "device.reconnect": lambda p: f"device:{p.get('name', '')}",
    "device.disconnect": lambda p: f"device:{p.get('name', '')}",
    "device.setup": lambda p: f"device:{p.get('name', '')}",
    "tab.run_start": lambda p: f"tab:{p.get('tab_id', '')}",
    "tab.analyze": lambda p: f"analyze:{p.get('tab_id', '')}",
    "tab.post_analyze": lambda p: f"post_analyze:{p.get('tab_id', '')}",
}

OPERATION_KEY_OF: Mapping[str, OperationKeyFn] = MappingProxyType(
    _OPERATION_KEY_OF_DATA
)


@dataclass(frozen=True)
class MeasureMcpPolicy:
    """Immutable policy tables used by ``MeasureMcpSession``."""

    guard_deps: VersionPatternMap
    read_reveals: VersionPatternMap
    operation_key_of: Mapping[str, OperationKeyFn]


DEFAULT_POLICY = MeasureMcpPolicy(
    guard_deps=GUARD_DEPS,
    read_reveals=READ_REVEALS,
    operation_key_of=OPERATION_KEY_OF,
)


def expand_pattern_keys(
    patterns: tuple[str, ...], params: dict[str, Any], source_table: Mapping[str, int]
) -> dict[str, int]:
    """Expand version-key patterns against ``params`` and ``source_table``."""

    out: dict[str, int] = {}
    for pattern in patterns:
        if pattern == "device:*":
            for key, version in source_table.items():
                if key.startswith("device:"):
                    out[key] = version
            continue
        key = pattern.format(
            tab_id=params.get("tab_id", ""),
            editor_id=params.get("editor_id", ""),
            name=params.get("name", ""),
        )
        out[key] = source_table.get(key, 0)
    return out


def describe_stale_keys(keys: list[Any]) -> list[str]:
    """Translate stale resource keys into agent-facing phrases."""

    out: list[str] = []
    for raw in keys:
        key = str(raw)
        if key == "context":
            out.append("the active context (md/ml)")
        elif key == "soc":
            out.append("the SoC connection")
        elif key == "devices:__set__":
            out.append("the set of devices (one added/removed)")
        elif key == "arb_waveforms":
            out.append("the arbitrary waveform asset store")
        elif key.startswith("device:"):
            out.append(f"device {key[len('device:') :]!r}")
        elif key.startswith("editor:"):
            out.append("the cfg-editor draft")
        elif key.startswith("tab:"):
            facet = key.split(":", 2)[2] if key.count(":") >= 2 else ""
            label = {
                "cfg": "this tab's cfg",
                "result": "this tab's run result",
                "analyze": "this tab's analysis",
                "save_path": "this tab's save path",
            }.get(facet, "this tab")
            if label not in out:
                out.append(label)
        else:
            out.append(key)
    return out
