"""Human-readable descriptions of stale GUI resource keys."""

from __future__ import annotations

from typing import Any


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
