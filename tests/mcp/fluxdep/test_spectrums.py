"""Agent spectrum decisions compose native RPCs and return bounded summaries."""

import base64
from io import BytesIO

import pytest
from PIL import Image
from zcu_tools.mcp.core.reply import ToolReply

from tests.mcp.fluxdep._support import Client, GuiResponse


@pytest.fixture
def snapshot() -> dict[str, object]:
    """Native complete snapshot with unsorted points and independent extents."""
    return {
        "name": "raw.hdf5", "exists": True, "spec_type": "TwoTone",
        "aligned": True, "points_completed": True, "pending": "sample",
        "flux_half": 1.23456, "flux_int": 3.23456, "flux_period": 4.0,
        "alignment": {"inherited_from": "seed.hdf5"},
        "raw_axes": {
            "dev_values": {"count": 5, "minimum": -1.23456, "maximum": 3.45678, "unit": "native"},
            "freqs": {"count": 6, "minimum": 4.12345, "maximum": 5.98765, "unit": "GHz"},
        },
        "points": {"dev_values": [1.1, 0.2, 0.4], "fluxs": [0.53712, 0.20821, 0.3], "freqs": [5.97011, 3.58012, 4.2]},
    }


@pytest.fixture
def image_reply() -> dict[str, object]:
    """Small valid producer PNG; no backend or file mutation."""
    buffer = BytesIO()
    with Image.new("RGB", (12, 8), color="red") as image:
        image.save(buffer, format="PNG")
    png = buffer.getvalue()
    return {"png_b64": base64.b64encode(png).decode("ascii"), "bytes": len(png)}


@pytest.mark.parametrize("load", [False, True])
def test_snapshot_and_load_return_ranges_and_native_evidence(
    client: Client, snapshot: dict[str, object], image_reply: dict[str, object], load: bool,
) -> None:
    responses = [GuiResponse(result=snapshot), GuiResponse(result=image_reply)]
    if load:
        responses.insert(0, GuiResponse(result={"name": "raw.hdf5"}))
    client.transport.replies.extend(responses)
    tool = "fluxdep_spectrum_load" if load else "fluxdep_spectrum_snapshot"
    args = {"filepath": "/raw.hdf5", "spec_type": "TwoTone", "inherit_from": "seed.hdf5", "transpose_axes": True} if load else {"name": "raw.hdf5"}
    result = client.server.tools[tool]["handler"](args)
    assert isinstance(result, ToolReply)
    assert result.data == {
        "name": "raw.hdf5", "type": "TwoTone", "stage": "sampled", "pending": "sample",
        "axes": {
            "dev": {"count": 5, "min": -1.235, "max": 3.457, "unit": "native"},
            "freq": {"count": 6, "min": 4.123, "max": 5.988, "unit": "GHz"},
        },
        "alignment": {"flux_half": 1.235, "flux_int": 3.235, "flux_period": 4.0, "inherited_from": "seed.hdf5"},
        "points": {"count": 3, "flux_range": [0.208, 0.537], "freq_range": [3.58, 5.97]},
    }
    assert len(result.images) == 1
    calls = [(r["method"], r["params"]) for r in client.transport.requests]
    assert calls == ([("spectrum.load", args)] if load else []) + [
        ("spectrum.snapshot", {"name": "raw.hdf5"}),
        ("spectrum.render", {"name": "raw.hdf5"}),
    ]


@pytest.mark.parametrize("selected", [None, [True, False, True]])
def test_list_summarizes_stages_filter_and_stale_fit(client: Client, selected: list[bool] | None) -> None:
    client.transport.replies.extend([
        GuiResponse(result={"spectrums": [
            {"name": "a", "spec_type": "TwoTone", "aligned": True, "points_completed": True, "point_count": 3, "pending": None},
            {"name": "b", "spec_type": "OneTone", "aligned": True, "points_completed": False, "point_count": 0, "pending": "sample"},
            {"name": "c", "spec_type": "TwoTone", "aligned": False, "points_completed": False, "point_count": 0, "pending": "align"},
        ]}),
        GuiResponse(result={"selected": selected, "min_distance": 0.1, "pending": True}),
        GuiResponse(result={"has_result": True, "stale": True, "stale_reasons": ["filter changed"], "params": {"EJ": 3.0}, "quality": {"residuals": [1.0, 2.0]}}),
    ])
    result = client.server.tools["fluxdep_spectrum_list"]["handler"]({})
    assert isinstance(result, ToolReply)
    assert result.data == {
        "spectrums": [
            {"name": "a", "type": "TwoTone", "stage": "sampled", "point_count": 3, "pending": None},
            {"name": "b", "type": "OneTone", "stage": "aligned", "point_count": 0, "pending": "sample"},
            {"name": "c", "type": "TwoTone", "stage": "loaded", "point_count": 0, "pending": "align"},
        ],
        "filter": {"selected": 3 if selected is None else 2, "total": 3, "pending": True},
        "fit": {"has_result": True, "stale": True, "stale_reasons": ["filter changed"]},
    }
    assert [r["method"] for r in client.transport.requests] == ["spectrum.list", "selection.snapshot", "fit.result"]


def test_remove_returns_native_receipt_without_followup_reads(client: Client) -> None:
    client.transport.replies.append(GuiResponse(result={"name": "a", "removed": True}))
    result = client.server.tools["fluxdep_spectrum_remove"]["handler"]({"name": "a"})
    assert isinstance(result, ToolReply)
    assert result.data == {"name": "a", "removed": True}
    assert [(r["method"], r["params"]) for r in client.transport.requests] == [("spectrum.remove", {"name": "a"})]
