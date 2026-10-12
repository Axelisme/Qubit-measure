"""Filter publication and native evidence through public owner and RPC seams."""

import base64
import io
from collections.abc import Mapping

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from PIL import Image
from zcu_tools.gui.app.fluxdep.event_bus import InteractiveChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _context(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    context = reply["result"]["context"]
    assert isinstance(context, dict)
    return context


def _state(context: Mapping[str, object]) -> dict[str, object]:
    state = context["state"]
    assert isinstance(state, dict)
    return state


def _joint_command(
    harness: RouteHarness,
    link: ClientLink,
    context: Mapping[str, object],
    command: str,
    **params: object,
) -> RouteReply:
    return harness.request(
        link,
        "selection.interactive.command",
        context_id=context["context_id"],
        command=command,
        params=params,
    )


def test_filter_finish_invalidated_is_native_fit_dependency(
    joint_route: RouteHarness,
) -> None:
    link = joint_route.client()
    context = _context(joint_route.request(link, "selection.interactive.open"))
    finished = _joint_command(joint_route, link, context, "finish")
    assert finished["ok"] and "result" in finished
    assert finished["result"].get("invalidated") == ["fit"]


def test_filter_finish_publishes_closes_and_reports_each_spectrum(
    joint_route: RouteHarness,
):
    link = joint_route.client()
    context = _context(joint_route.request(link, "selection.interactive.open"))
    assert _joint_command(
        joint_route,
        link,
        context,
        "stroke",
        vertices=[[0.5, 4.5]],
        width=0.03,
        mode="erase",
    )["ok"]
    finished = _joint_command(joint_route, link, context, "finish")
    assert finished["ok"] is True, finished
    result = _context(finished)
    assert result["closed"] is True
    assert result["can_undo"] is False
    assert _state(result)["per_spectrum"] == {"a": [2, 3], "empty": [0, 0], "b": [0, 1]}
    np.testing.assert_array_equal(
        joint_route.ctrl.state.selection_mask(), [True, False, True, False]
    )
    assert joint_route.ctrl.interactive.inspect("selection") is None
    assert joint_route.request(link, "selection.render")["ok"]
    assert joint_route.ctrl.interactive.inspect("selection") is None


def test_published_filter_render_is_readonly_without_a_draft(joint_route: RouteHarness):
    link = joint_route.client()
    context = _context(joint_route.request(link, "selection.interactive.open"))
    assert _joint_command(
        joint_route,
        link,
        context,
        "stroke",
        vertices=[[0.5, 4.5]],
        width=0.03,
        mode="erase",
    )["ok"]
    # The finish promise is covered separately; publish/cancel reaches the same
    # published input without making this render regression depend on finish.
    assert _joint_command(joint_route, link, context, "apply")["ok"]
    assert _joint_command(joint_route, link, context, "cancel")["ok"]
    observer = joint_route.client(mode="readonly")
    before = joint_route.ctrl.state.version.snapshot()
    rendered = joint_route.request(observer, "selection.render")
    assert rendered["ok"] is True, rendered
    encoded = rendered["result"]["png_b64"]
    assert isinstance(encoded, str)
    png = base64.b64decode(encoded)
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
    assert joint_route.ctrl.interactive.inspect("selection") is None
    assert joint_route.ctrl.state.version.snapshot() == before


def test_published_filter_figure_marks_masks_without_reading_live_draft(
    joint_route: RouteHarness,
):
    joint_route.ctrl.set_selection(np.array([True, False, True, False]), 0.025)
    link = joint_route.client()
    context = _context(joint_route.request(link, "selection.interactive.open"))
    # A fresh Filter starts all-selected; rendering must still use publication.
    assert _joint_command(
        joint_route,
        link,
        context,
        "stroke",
        vertices=[[0.0, 4.0]],
        width=0.03,
        mode="erase",
    )["ok"]
    figure = Figure()
    joint_route.ctrl.render_selection(figure)
    scatter = [
        artist
        for artist in figure.axes[0].collections
        if isinstance(artist, PathCollection)
    ]
    np.testing.assert_allclose(np.asarray(scatter[0].get_offsets()), [[0.5, 4.5]])
    np.testing.assert_allclose(
        np.asarray(scatter[1].get_offsets()), [[0.0, 4.0], [1.0, 5.0]]
    )
    np.testing.assert_allclose(np.asarray(scatter[-2].get_offsets()), [[0.5, 4.5]])
    assert joint_route.ctrl.interactive.inspect("selection") is not None


def test_filter_finish_retains_receipt_without_closing_reentrant_successor(
    joint_route: RouteHarness,
):
    link = joint_route.client()
    context = _context(joint_route.request(link, "selection.interactive.open"))
    replaced = False

    def reopen(payload: InteractiveChangedPayload, _meta: EventMeta) -> None:
        nonlocal replaced
        if payload.kind == "selection" and payload.phase == "updated" and not replaced:
            replaced = True
            joint_route.ctrl.interactive.cancel("selection")
            joint_route.ctrl.interactive.begin_cross_selection()

    subscription = joint_route.ctrl.bus.subscribe_with_meta(
        InteractiveChangedPayload, reopen
    )
    try:
        finished = _joint_command(joint_route, link, context, "finish")
    finally:
        subscription.unsubscribe()
    assert _context(finished)["closed"] is True
    successor = joint_route.ctrl.interactive.inspect("selection")
    assert successor is not None
    assert successor.context_id != context["context_id"]
