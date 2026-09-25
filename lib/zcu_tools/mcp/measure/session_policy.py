"""Pure helpers for GUI-owned version-key patterns and stale descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def expand_pattern_keys(
    patterns: tuple[str, ...], params: dict[str, Any], source_table: Mapping[str, int]
) -> dict[str, int]:
    """Expand the catalog's resource patterns against observed versions."""
    out: dict[str, int] = {}
    for pattern in patterns:
        if pattern == "device:*":
            for key, version in source_table.items():
                if key.startswith("device:"):
                    out[key] = version
            continue
        writeback_resource = {
            "analysis": "analyze",
            "post_analysis": "post_analyze",
        }.get(str(params.get("subtab_id", "")), "invalid_writeback_subtab")
        key = pattern.format(
            tab_id=params.get("tab_id", ""),
            editor_id=params.get("editor_id", ""),
            name=params.get("name", ""),
            writeback_resource=writeback_resource,
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
                "post_analyze": "this tab's post-analysis",
                "save_path": "this tab's save path",
                "path:data": "this tab's data path",
                "path:analysis_image": "this tab's analysis image path",
                "path:post_analysis_image": "this tab's post-analysis image path",
            }.get(facet, "this tab")
            if label not in out:
                out.append(label)
        else:
            out.append(key)
    return out
